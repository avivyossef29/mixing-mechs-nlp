#!/usr/bin/env python3
"""
Run SSM-state patching experiments across all layers of a Mamba2 model.

Instead of patching the residual stream (as in run_layer_experiments.py),
this script patches the SSM recurrent state h inside the Mamba2Mixer.
This tests whether binding information is stored in the SSM state rather
than (or in addition to) the residual stream.

Approach:
1. Run the CF prompt through the model, hooking each Mamba2Mixer's torch_forward
   to capture the intermediate SSM 'states' tensor at the target layer.
2. Run the normal prompt, but at the target layer, inject the CF's SSM states
   into the computation, replacing the normal states with the CF states.
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from contextlib import contextmanager
from functools import partial

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append("CausalAbstraction")

# Suppress transformers logging
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
import transformers

transformers.logging.set_verbosity_error()

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from training import (
    sample_answerable_question_template,
    get_counterfactual_datasets,
    ppkn_simpler_counterfactual_template_split_key_loc,
)
from grammar.task_to_causal_model import multi_order_multi_schema_task_to_lookbacks_generic_causal_model
from grammar.schemas import SCHEMA_BOXES
from tasks.dist import (
    get_end_str,
    format_prompt,
    to_str_tokens,
    try_schema_checker,
    _num_layers,
)
from plotting import plot_patch_effect

# Import Mamba2-specific modules
from transformers.models.mamba2.modeling_mamba2 import (
    Mamba2Mixer,
    apply_mask_to_padding_states,
    pad_tensor_by_size,
    reshape_into_chunks,
    segment_sum,
)


# ---------------------------------------------------------------------------
# SSM state capture / injection via monkey-patching torch_forward
# ---------------------------------------------------------------------------

def _patched_torch_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    # --- extra args injected via partial ---
    capture_dict=None,       # if not None, store {"states": ..., "ssm_state": ...}
    inject_states=None,      # if not None, a tensor to replace 'states' with
):
    """
    A modified copy of Mamba2Mixer.torch_forward that optionally
    captures or injects the SSM chunk-boundary states.
    """
    batch_size, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype

    # 1. Gated MLP's linear projection
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    projected_states = self.in_proj(hidden_states)
    d_mlp = (
        projected_states.shape[-1]
        - 2 * self.intermediate_size
        - 2 * self.n_groups * self.ssm_state_size
        - self.num_heads
    ) // 2
    _, _, gate, hidden_states_B_C, dt = projected_states.split(
        [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
    )

    # 2. Convolution sequence transformation
    if cache_params is not None and cache_position is not None and cache_position[0] > 0:
        cache_params.update_conv_state(
            layer_idx=self.layer_idx, new_conv_state=hidden_states_B_C, cache_init=False
        )
        conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)
        hidden_states_B_C = torch.sum(conv_states * self.conv1d.weight.squeeze(1), dim=-1)
        if self.use_conv_bias:
            hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
        hidden_states_B_C = self.act(hidden_states_B_C)
    else:
        if cache_params is not None:
            hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
            conv_states_pad = nn.functional.pad(
                hidden_states_B_C_transposed,
                (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
            )
            cache_params.update_conv_state(
                layer_idx=self.layer_idx, new_conv_state=conv_states_pad, cache_init=True
            )
        hidden_states_B_C = self.act(
            self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        )

    hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
    hidden_states, B, C = torch.split(
        hidden_states_B_C,
        [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
        dim=-1,
    )

    # 3. SSM transformation
    A = -torch.exp(self.A_log.float())  # [num_heads]

    # -- begin SSD naive (no einsums) --
    dt = nn.functional.softplus(dt + self.dt_bias)
    dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
    hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
    B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
    C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
    B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
    C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
    pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

    D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

    # Discretize x and A
    hidden_states = hidden_states * dt[..., None]
    A = A.to(hidden_states.dtype) * dt

    # Rearrange into blocks/chunks
    hidden_states, A, B, C = [
        reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)
    ]

    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segment_sum(A))

    G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
    G = G_intermediate.sum(dim=-1)

    M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    M = M_intermediate.sum(dim=-1)

    Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    # 2. Compute the state for each intra-chunk (B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

    # 3. Compute the inter-chunk SSM recurrence
    if cache_params is not None and cache_position is not None and cache_position[0] > 0:
        previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=states.device)
    else:
        previous_states = torch.zeros_like(states[:, :1])
    states = torch.cat([previous_states, states], dim=1)
    decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, ssm_state = new_states[:, :-1], new_states[:, -1]

    # ============================================================
    # CAPTURE: save the SSM states if requested
    # ============================================================
    if capture_dict is not None:
        capture_dict["states"] = states.detach().clone()
        capture_dict["ssm_state"] = ssm_state.detach().clone()

    # ============================================================
    # INJECT: replace SSM states with counterfactual states
    # ============================================================
    if inject_states is not None:
        cf_states = inject_states["states"].to(device=states.device, dtype=states.dtype)
        cf_ssm_state = inject_states["ssm_state"].to(device=ssm_state.device, dtype=ssm_state.dtype)
        # Match shapes (CF might have different padding/chunks)
        if cf_states.shape == states.shape:
            states = cf_states
        else:
            # Use the minimum number of chunks
            min_chunks = min(states.shape[1], cf_states.shape[1])
            states[:, :min_chunks] = cf_states[:, :min_chunks]
        if cf_ssm_state.shape == ssm_state.shape:
            ssm_state = cf_ssm_state

    # 4. Compute state -> output conversion per chunk (C terms)
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C[..., None, :] * states[:, :, None, ...]
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

    # Add output of intra-chunk and inter-chunk terms
    y = Y_diag + Y_off
    y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

    y = y + D_residual
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    y = y.reshape(batch_size, seq_len, -1)

    # Init cache
    if ssm_state is not None and cache_params is not None:
        cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

    scan_output = self.norm(y, gate)

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.to(dtype))
    return contextualized_states


# ---------------------------------------------------------------------------
# Context manager for SSM-state patching
# ---------------------------------------------------------------------------

@contextmanager
def run_with_cf_ssm_state(
    model,
    tokenizer,
    normal_str,
    cf_str,
    layer_idx=18,
    device=None,
):
    """
    Patch the SSM recurrent state at block `layer_idx` with
    counterfactual SSM states taken from `cf_str`.

    1. Monkey-patch the target layer's mixer to use our custom torch_forward
    2. Run CF prompt to capture SSM states
    3. Run normal prompt with CF's SSM states injected
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    # Get the target mixer
    target_block = model.backbone.layers[layer_idx]
    mixer = target_block.mixer

    # Save original forward
    original_forward = mixer.forward

    # --- Step 1: Run CF prompt to capture SSM states ---
    cf_capture = {}

    def cf_forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        return _patched_torch_forward(
            mixer, hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
            capture_dict=cf_capture,
            inject_states=None,
        )

    mixer.forward = cf_forward

    enc_cf = tokenizer(cf_str, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**enc_cf)

    # --- Step 2: Set up injection for normal prompt ---
    def inject_forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        return _patched_torch_forward(
            mixer, hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
            capture_dict=None,
            inject_states=cf_capture,
        )

    mixer.forward = inject_forward

    try:
        yield  # inside the with-block, call model on 'normal_str'
    finally:
        mixer.forward = original_forward
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Experiment logic (same structure as run_layer_experiments.py)
# ---------------------------------------------------------------------------

def get_schema_by_name(schema_name: str):
    """Get schema object by name."""
    from grammar.schemas import (
        SCHEMA_FILLING_LIQUIDS,
        SCHEMA_PEOPLE_AND_OBJECTS,
        SCHEMA_PROGRAMMING_PEOPLE_DICT,
        SCHEMA_MUSIC_PERFORMANCE,
        SCHEMA_LAB_EXPERIMENTS,
        SCHEMA_CHEMISTRY_EXPERIMENTS,
        SCHEMA_TRANSPORTATION,
        SCHEMA_SPORTS_EVENTS,
        SCHEMA_SPACE_OBSERVATIONS,
        SCHEMA_BOXES,
    )
    schemas = {
        "SCHEMA_FILLING_LIQUIDS": SCHEMA_FILLING_LIQUIDS,
        "SCHEMA_PEOPLE_AND_OBJECTS": SCHEMA_PEOPLE_AND_OBJECTS,
        "SCHEMA_PROGRAMMING_PEOPLE_DICT": SCHEMA_PROGRAMMING_PEOPLE_DICT,
        "SCHEMA_MUSIC_PERFORMANCE": SCHEMA_MUSIC_PERFORMANCE,
        "SCHEMA_LAB_EXPERIMENTS": SCHEMA_LAB_EXPERIMENTS,
        "SCHEMA_CHEMISTRY_EXPERIMENTS": SCHEMA_CHEMISTRY_EXPERIMENTS,
        "SCHEMA_TRANSPORTATION": SCHEMA_TRANSPORTATION,
        "SCHEMA_SPORTS_EVENTS": SCHEMA_SPORTS_EVENTS,
        "SCHEMA_SPACE_OBSERVATIONS": SCHEMA_SPACE_OBSERVATIONS,
        "SCHEMA_BOXES": SCHEMA_BOXES,
    }
    if schema_name not in schemas:
        raise ValueError(f"Unknown schema name: {schema_name}. Available: {list(schemas.keys())}")
    return schemas[schema_name]


def sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def run_experiment_for_layer(
    model,
    tokenizer,
    train_ds,
    schema,
    num_instances: int,
    num_samples: int,
    layer: int,
    cat_to_query: int,
    model_id: str,
    generate: bool = False,
):
    results = {
        "normal": [],
        "cf": [],
        "source_pos": [],
        "positional_index": [],
        "keyload_index": [],
        "payload_index": [],
        "layer": [],
        "prediction": [],
        "positional_prediction": [],
        "payload_prediction": [],
        "keyload_prediction": [],
        "no_effect_prediction": [],
        "patch_effect": [],
        "dist": [],
        "distance": [],
        "generated": [],
    }

    train = train_ds[schema.name][schema.name]
    token_positions = [-1]
    end_str = get_end_str(model_id)
    model_id_str = model_id

    for cur_index in tqdm(range(num_samples), desc=f"Layer {layer}"):
        prompt = format_prompt(tokenizer, train[cur_index]["input"]["raw_input"])
        cf_prompt = format_prompt(tokenizer, train[cur_index]["counterfactual_inputs"][0]["raw_input"])
        prompt_str_tokenized = to_str_tokens(tokenizer, prompt)
        metadata = train[cur_index]["input"]["metadata"]

        answer_indices = []
        keyload_index = None
        payload_index = None
        for i, token in enumerate(prompt_str_tokenized):
            if "qwen" in model_id_str.lower() and i < 10:
                continue

            if schema.matchers[cat_to_query](token):
                answer_indices.append(i)

                if prompt_str_tokenized[i].lower().strip() in metadata["keyload"].lower().strip():
                    keyload_index = len(answer_indices) - 1

                if prompt_str_tokenized[i].lower().strip() in metadata["payload"].lower().strip():
                    payload_index = len(answer_indices) - 1

        assert (
            len(answer_indices) == num_instances
        ), f"Expected {num_instances} answer indices, got {len(answer_indices)}.\nPrompt_str_tokenized: {prompt_str_tokenized}.\n{[prompt_str_tokenized[i] for i in answer_indices]}."
        assert (
            keyload_index is not None
        ), f"Keyload [{metadata['keyload']}] index is None. Prompt_str_tokenized: {prompt_str_tokenized}.\n{[prompt_str_tokenized[i] for i in answer_indices]}."
        assert (
            payload_index is not None
        ), f"Payload [{metadata['payload']}] index is None. Prompt_str_tokenized: {prompt_str_tokenized}.\n{[prompt_str_tokenized[i] for i in answer_indices]}."

        pos_index = metadata["dst_index"]

        # Use SSM state patching instead of residual stream patching
        with run_with_cf_ssm_state(
            model, tokenizer, prompt, cf_prompt, layer_idx=layer
        ):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                logits = model(input_ids).logits

            token_ids_at_answer_positions = input_ids[0, answer_indices].tolist()
            values = logits[0, -1, token_ids_at_answer_positions]

            pos_pred = values.argmax().item()

            if generate:
                pred_ids = model.generate(input_ids, max_new_tokens=schema.max_new_tokens, do_sample=False)
                pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                pred = pred[pred.find(end_str) + len(end_str) :]
            else:
                pred = prompt_str_tokenized[answer_indices[pos_pred]]

            pred = re.sub(r'\s+', '', pred.strip())

            if try_schema_checker(pred, metadata["positional"], schema):
                patch_effect = "positional"
            elif try_schema_checker(pred, metadata["keyload"], schema):
                patch_effect = "lexical"
            elif try_schema_checker(pred, metadata["payload"], schema):
                patch_effect = "reflexive"
            elif try_schema_checker(pred, metadata["no_effect"], schema):
                patch_effect = "no_effect"
            else:
                patch_effect = "mixed"

            results["normal"].append(prompt)
            results["cf"].append(cf_prompt)
            results["source_pos"].append(metadata["src_positional_index"])
            results["positional_index"].append(pos_index)
            results["keyload_index"].append(keyload_index)
            results["payload_index"].append(payload_index)
            results["layer"].append(layer)
            results["positional_prediction"].append(metadata["positional"])
            results["payload_prediction"].append(metadata["payload"])
            results["keyload_prediction"].append(metadata["keyload"])
            results["no_effect_prediction"].append(metadata["no_effect"])
            results["patch_effect"].append(patch_effect)
            results["prediction"].append(pred)
            results["dist"].append(values.tolist())
            results["distance"].append(pos_index - pos_pred)
            results["generated"].append(generate)

    df = pd.DataFrame(results)
    return df


def save_results(df: pd.DataFrame, fig, output_dir: Path, layer: int):
    layer_dir = output_dir / f"layer_{layer}"
    layer_dir.mkdir(exist_ok=True)

    csv_path = layer_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    png_path = layer_dir / "patch_effect.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  Saved PNG: {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SSM-state patching experiments across all layers of a Mamba2 model"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="AntonV/mamba2-2.7b-hf",
        help="Mamba2 model ID from HuggingFace (default: AntonV/mamba2-2.7b-hf)",
    )
    parser.add_argument("--num-instances", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--cat-to-query", type=int, default=1)
    parser.add_argument("--schema-name", type=str, default="SCHEMA_BOXES")
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-layer", type=int, default=None)
    parser.add_argument("--end-layer", type=int, default=None)

    args = parser.parse_args()

    schema = get_schema_by_name(args.schema_name)
    print(f"[+] Using schema: {args.schema_name}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name = sanitize_model_name(args.model_id)
        output_dir = Path(f"{model_name}_ssm_patch")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"[+] Output directory: {output_dir.absolute()}")

    print(f"[+] Loading model: {args.model_id}")
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    tokenizer_kwargs = {"token": args.hf_token} if args.hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tokenizer_kwargs)

    num_layers = _num_layers(model)
    print(f"[+] Model has {num_layers} layers")
    print(f"[+] SSM state patching mode (patching h, not residual stream)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"[+] Random seed set to: {args.seed}")

    print(f"[+] Setting up causal model and datasets")
    causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
        [schema], args.num_instances, num_fillers_per_item=0, fillers=False
    )
    causal_models = {schema.name: causal_model}

    counterfactual_template = ppkn_simpler_counterfactual_template_split_key_loc

    train_ds, test_ds, fps = get_counterfactual_datasets(
        None,
        [schema],
        num_samples=args.num_samples,
        num_instances=args.num_instances,
        cat_indices_to_query=[0],
        answer_cat_id=args.cat_to_query,
        do_assert=True,
        do_filter=False,
        counterfactual_template=counterfactual_template,
        causal_models=causal_models,
        sample_an_answerable_question=sample_answerable_question_template,
    )

    start_layer = args.start_layer if args.start_layer is not None else 0
    end_layer = args.end_layer if args.end_layer is not None else (num_layers - 1)

    if start_layer < 0:
        raise ValueError(f"start-layer must be >= 0, got {start_layer}")
    if end_layer >= num_layers:
        raise ValueError(f"end-layer must be < {num_layers}, got {end_layer}")
    if start_layer > end_layer:
        raise ValueError(f"start-layer ({start_layer}) must be <= end-layer ({end_layer})")

    layer_range = range(start_layer, end_layer + 1)
    num_layers_to_process = len(layer_range)

    print(f"[+] Running SSM-state patching for layers {start_layer} to {end_layer} ({num_layers_to_process} layers)...")
    for layer in tqdm(layer_range, desc="Layers"):
        print(f"\n[+] Processing layer {layer}/{num_layers - 1}")

        df = run_experiment_for_layer(
            model,
            tokenizer,
            train_ds,
            schema,
            args.num_instances,
            args.num_samples,
            layer,
            args.cat_to_query,
            args.model_id,
            generate=args.generate,
        )

        fig, ax = plot_patch_effect(df, include_reflexive=True, highest_near_pos=0)

        save_results(df, fig, output_dir, layer)

        plt.close(fig)

    print(f"\n[+] All experiments completed! Results saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
