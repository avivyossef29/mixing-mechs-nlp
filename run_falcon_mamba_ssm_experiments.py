#!/usr/bin/env python3
"""
Run SSM-state patching experiments across all layers of a Falcon-Mamba model.

Instead of patching the residual stream (as in run_layer_experiments.py),
this script patches the SSM recurrent state h inside the FalconMambaMixer.
This tests whether binding information is stored in the SSM state rather
than (or in addition to) the residual stream.

Falcon-Mamba uses Mamba1 architecture with a simple token-by-token recurrence:
    for i in range(seq_len):
        ssm_state = discrete_A[:,:,i,:] * ssm_state + deltaB_u[:,:,i,:]
        output_i  = ssm_state @ C[:,i,:]

We capture the full ssm_state trajectory from the CF prompt and inject it
during the normal prompt run.

Compatible with transformers 5.0.0 (cloud mamba2-gpu env).
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from contextlib import contextmanager

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
from tasks.dist import (
    get_end_str,
    format_prompt,
    to_str_tokens,
    try_schema_checker,
    _num_layers,
)
from plotting import plot_patch_effect


# ---------------------------------------------------------------------------
# Import Falcon-Mamba specific helpers
# ---------------------------------------------------------------------------
from transformers.models.falcon_mamba.modeling_falcon_mamba import (
    FalconMambaMixer,
    FalconMambaCache,
)

# rms_forward is a module-level function in falcon_mamba
try:
    from transformers.models.falcon_mamba.modeling_falcon_mamba import rms_forward
except ImportError:
    # Fallback: define it ourselves (same as in transformers source)
    def rms_forward(hidden_states, variance_epsilon=1e-6):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        return hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# SSM state capture / injection via monkey-patching slow_forward
# ---------------------------------------------------------------------------

def _patched_slow_forward(
    self,
    input_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    # --- extra args for capture/inject ---
    capture_dict=None,
    inject_states=None,
):
    """
    A modified copy of FalconMambaMixer.slow_forward that optionally
    captures or injects the SSM state at each token position.

    Only the CAPTURE and INJECT blocks are added. Everything else is
    identical to the original slow_forward from transformers 5.0.0.
    """
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    # 1. Gated MLP's linear projection
    projected_states = self.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 2. Convolution sequence transformation
    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone()
        ssm_state = ssm_state.to(hidden_states.device)
        if cache_position is not None and cache_position.shape[0] == self.conv_kernel_size:
            conv_state = nn.functional.pad(
                hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
            )
            cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        else:
            conv_state = cache_params.update_conv_state(
                self.layer_idx, hidden_states, cache_position
            )
            conv_state = conv_state.to(self.conv1d.weight.device)
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device,
            dtype=dtype,
        )
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    # 3. State Space Model sequence transformation
    # 3.a. Selection
    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters,
        [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
        dim=-1,
    )

    B = rms_forward(B, variance_epsilon=self.rms_eps)
    C = rms_forward(C, variance_epsilon=self.rms_eps)
    time_step = rms_forward(time_step, variance_epsilon=self.rms_eps)

    discrete_time_step = self.dt_proj(time_step)
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)

    # 3.b. Discretization
    A = -torch.exp(self.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # 3.c. Recurrence y ← SSM(A, B, C)(x)
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]

        # ==============================================================
        # INJECT: replace ssm_state with CF's ssm_state at this token
        # ==============================================================
        if inject_states is not None and i < inject_states.shape[2]:
            ssm_state = inject_states[:, :, i, :].to(
                device=ssm_state.device, dtype=ssm_state.dtype
            )

        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
        scan_outputs.append(scan_output[:, :, 0])

    # ==============================================================
    # CAPTURE: save the full ssm_state trajectory
    # ==============================================================
    # (We do a separate pass to capture if needed, to avoid
    #  storing during inject which would be the injected states)

    scan_output = torch.stack(scan_outputs, dim=-1)
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)

    if cache_params is not None:
        cache_params.update_ssm_state(self.layer_idx, ssm_state)

    # 4. Final linear projection
    contextualized_states = self.out_proj(scan_output.transpose(1, 2))
    return contextualized_states


def _capture_slow_forward(
    self,
    input_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    capture_dict=None,
):
    """
    Same as slow_forward but captures ssm_state at every token position.
    """
    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    projected_states = self.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    if cache_params is not None:
        ssm_state = cache_params.ssm_states[self.layer_idx].clone()
        ssm_state = ssm_state.to(hidden_states.device)
        if cache_position is not None and cache_position.shape[0] == self.conv_kernel_size:
            conv_state = nn.functional.pad(
                hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
            )
            cache_params.update_conv_state(self.layer_idx, conv_state, cache_position)
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        else:
            conv_state = cache_params.update_conv_state(
                self.layer_idx, hidden_states, cache_position
            )
            conv_state = conv_state.to(self.conv1d.weight.device)
            hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
            if self.use_conv_bias:
                hidden_states += self.conv1d.bias
            hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
    else:
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device,
            dtype=dtype,
        )
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters,
        [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
        dim=-1,
    )

    B = rms_forward(B, variance_epsilon=self.rms_eps)
    C = rms_forward(C, variance_epsilon=self.rms_eps)
    time_step = rms_forward(time_step, variance_epsilon=self.rms_eps)

    discrete_time_step = self.dt_proj(time_step)
    discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)

    A = -torch.exp(self.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    # Recurrence with capture
    scan_outputs = []
    ssm_states_trajectory = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        ssm_states_trajectory.append(ssm_state.detach().clone())
        scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
        scan_outputs.append(scan_output[:, :, 0])

    # CAPTURE: store trajectory [batch, intermediate_size, seq_len, ssm_state_size]
    if capture_dict is not None:
        capture_dict["ssm_states"] = torch.stack(ssm_states_trajectory, dim=2)

    scan_output = torch.stack(scan_outputs, dim=-1)
    scan_output = scan_output + (hidden_states * self.D[None, :, None])
    scan_output = scan_output * self.act(gate)

    if cache_params is not None:
        cache_params.update_ssm_state(self.layer_idx, ssm_state)

    contextualized_states = self.out_proj(scan_output.transpose(1, 2))
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
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    target_block = model.backbone.layers[layer_idx]
    mixer = target_block.mixer
    original_forward = mixer.forward

    # --- Step 1: Run CF prompt to capture SSM states ---
    cf_capture = {}

    def cf_forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        return _capture_slow_forward(
            mixer,
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
            capture_dict=cf_capture,
        )

    mixer.forward = cf_forward

    enc_cf = tokenizer(cf_str, return_tensors="pt").to(device)
    with torch.no_grad():
        model(**enc_cf)

    # --- Step 2: Set up injection for normal prompt ---
    cf_ssm_states = cf_capture["ssm_states"]  # [batch, intermediate_size, seq_len, ssm_state_size]

    def inject_forward(hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        return _patched_slow_forward(
            mixer,
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
            capture_dict=None,
            inject_states=cf_ssm_states,
        )

    mixer.forward = inject_forward

    try:
        yield
    finally:
        mixer.forward = original_forward
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Experiment logic (same structure as run_layer_experiments.py)
# ---------------------------------------------------------------------------

def get_schema_by_name(schema_name: str):
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
    end_str = get_end_str(model_id)
    model_id_str = model_id

    for cur_index in tqdm(range(num_samples), desc=f"Layer {layer}"):
        prompt = format_prompt(tokenizer, train[cur_index]["input"]["raw_input"])
        cf_prompt = format_prompt(
            tokenizer, train[cur_index]["counterfactual_inputs"][0]["raw_input"]
        )
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

                if (
                    prompt_str_tokenized[i].lower().strip()
                    in metadata["keyload"].lower().strip()
                ):
                    keyload_index = len(answer_indices) - 1

                if (
                    prompt_str_tokenized[i].lower().strip()
                    in metadata["payload"].lower().strip()
                ):
                    payload_index = len(answer_indices) - 1

        assert len(answer_indices) == num_instances, (
            f"Expected {num_instances} answer indices, got {len(answer_indices)}.\n"
            f"Prompt_str_tokenized: {prompt_str_tokenized}.\n"
            f"{[prompt_str_tokenized[i] for i in answer_indices]}."
        )
        assert keyload_index is not None, (
            f"Keyload [{metadata['keyload']}] index is None. "
            f"Prompt_str_tokenized: {prompt_str_tokenized}.\n"
            f"{[prompt_str_tokenized[i] for i in answer_indices]}."
        )
        assert payload_index is not None, (
            f"Payload [{metadata['payload']}] index is None. "
            f"Prompt_str_tokenized: {prompt_str_tokenized}.\n"
            f"{[prompt_str_tokenized[i] for i in answer_indices]}."
        )

        pos_index = metadata["dst_index"]

        with run_with_cf_ssm_state(
            model, tokenizer, prompt, cf_prompt, layer_idx=layer
        ):
            input_ids = (
                tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            )
            with torch.no_grad():
                logits = model(input_ids).logits

            token_ids_at_answer_positions = input_ids[0, answer_indices].tolist()
            values = logits[0, -1, token_ids_at_answer_positions]

            pos_pred = values.argmax().item()

            if generate:
                pred_ids = model.generate(
                    input_ids,
                    max_new_tokens=schema.max_new_tokens,
                    do_sample=False,
                )
                pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                pred = pred[pred.find(end_str) + len(end_str) :]
            else:
                pred = prompt_str_tokenized[answer_indices[pos_pred]]

            pred = re.sub(r"\s+", "", pred.strip())

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
        description="Run SSM-state patching experiments across all layers of a Falcon-Mamba model"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="tiiuae/falcon-mamba-7b-instruct",
        help="Falcon-Mamba model ID (default: tiiuae/falcon-mamba-7b-instruct)",
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

    # Falcon-Mamba-7B needs 8-bit to fit on RTX 2080 (8GB)
    if "falcon" in args.model_id.lower() and torch.cuda.is_available():
        try:
            import bitsandbytes
            print(f"[+] Loading in 8-bit for GPU memory efficiency")
            model_kwargs.update({
                "load_in_8bit": True,
                "torch_dtype": torch.float16,
            })
        except ImportError:
            print(f"[+] bitsandbytes not available, loading in bfloat16")
    elif "falcon" in args.model_id.lower() and not torch.cuda.is_available():
        print(f"[+] No GPU detected, loading in bfloat16 (CPU mode)")
        model_kwargs["device_map"] = None

    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    tokenizer_kwargs = {"token": args.hf_token} if args.hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tokenizer_kwargs)

    num_layers = _num_layers(model)
    print(f"[+] Model has {num_layers} layers")
    print(f"[+] SSM state patching mode (patching h, not residual stream)")
    print(f"[+] Architecture: Falcon-Mamba (Mamba1, token-by-token recurrence)")

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
        raise ValueError(
            f"start-layer ({start_layer}) must be <= end-layer ({end_layer})"
        )

    layer_range = range(start_layer, end_layer + 1)
    num_layers_to_process = len(layer_range)

    print(
        f"[+] Running SSM-state patching for layers {start_layer} to {end_layer} "
        f"({num_layers_to_process} layers)..."
    )
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

    print(
        f"\n[+] All experiments completed! Results saved in: {output_dir.absolute()}"
    )


if __name__ == "__main__":
    main()
