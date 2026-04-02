#!/usr/bin/env python3
"""
Run ablation experiments on Falcon-H1 (hybrid Mamba+Attention model).

Falcon-H1 has parallel Mamba and Attention in every layer:
    output = residual + mamba_out * ssm_multiplier + attn_out * attn_multiplier + MLP

This script runs the standard residual stream patching experiment (like Yoav's)
but with an ablation: at specified layers, we zero out either the mamba or
attention contribution to isolate which mechanism drives entity binding.

Three modes:
  --ablate none:      Normal patching (baseline, same as original experiment)
  --ablate attention: Disrupt attention output at specified layers
  --ablate mamba:     Disrupt mamba/SSM output at specified layers

Two ablation methods:
  --ablate-method zero:  Zero out the output entirely
  --ablate-method noise: Add Gaussian noise (scale * output.std()) to corrupt the signal

This answers: "When we see the binding mechanism kick in at layer L,
is it the attention or the mamba component that's responsible?"
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path
from contextlib import contextmanager

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append("CausalAbstraction")

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
from grammar.task_to_causal_model import (
    multi_order_multi_schema_task_to_lookbacks_generic_causal_model,
)
from tasks.dist import (
    get_end_str,
    format_prompt,
    to_str_tokens,
    run_with_cf_hf,
    try_schema_checker,
    _num_layers,
)
from plotting import plot_patch_effect


# ---------------------------------------------------------------------------
# Ablation hooks: zero out attention or mamba at specific layers
# ---------------------------------------------------------------------------


def _install_ablation_hooks(model, ablate_mode, ablate_layers=None, method="zero", noise_scale=5.0):
    """
    Install hooks that disrupt either the attention or mamba output
    at specified layers.

    Args:
        model: FalconH1ForCausalLM
        ablate_mode: "attention" or "mamba"
        ablate_layers: list of layer indices to ablate (None = all layers)
        method: "zero" to zero out, "noise" to add scaled Gaussian noise
        noise_scale: multiplier for noise std (only used when method="noise")

    Returns:
        list of hook handles (to remove later)
    """
    handles = []
    num_layers = len(model.model.layers)

    if ablate_layers is None:
        ablate_layers = list(range(num_layers))

    for layer_idx in ablate_layers:
        layer = model.model.layers[layer_idx]

        if ablate_mode == "attention":
            target = layer.self_attn
        elif ablate_mode == "mamba":
            target = layer.mamba
        else:
            continue

        if method == "zero":
            def make_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)
                return hook
        else:  # noise
            def make_hook(scale=noise_scale):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        noise = torch.randn_like(output[0]) * output[0].std() * scale
                        return (output[0] + noise,) + output[1:]
                    noise = torch.randn_like(output) * output.std() * scale
                    return output + noise
                return hook

        h = target.register_forward_hook(make_hook())
        handles.append(h)

    return handles


@contextmanager
def ablation_context(model, ablate_mode, ablate_layers=None, method="zero", noise_scale=5.0):
    """Context manager that installs and removes ablation hooks."""
    if ablate_mode == "none":
        yield
        return

    handles = _install_ablation_hooks(model, ablate_mode, ablate_layers, method, noise_scale)
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Experiment logic
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
        raise ValueError(
            f"Unknown schema name: {schema_name}. Available: {list(schemas.keys())}"
        )
    return schemas[schema_name]


def sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def run_experiment_for_layer(
    model,
    tokenizer,
    train_ds,
    schema,
    num_instances,
    num_samples,
    layer,
    cat_to_query,
    model_id,
    ablate_mode="none",
    ablate_layers=None,
    ablate_method="zero",
    noise_scale=5.0,
    generate=False,
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
        prompt = format_prompt(
            tokenizer, train[cur_index]["input"]["raw_input"]
        )
        cf_prompt = format_prompt(
            tokenizer,
            train[cur_index]["counterfactual_inputs"][0]["raw_input"],
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

        assert len(answer_indices) == num_instances
        assert keyload_index is not None
        assert payload_index is not None

        pos_index = metadata["dst_index"]

        # Standard residual stream patching + ablation
        with ablation_context(model, ablate_mode, ablate_layers, ablate_method, noise_scale):
            with run_with_cf_hf(
                model,
                tokenizer,
                prompt,
                cf_prompt,
                layer_idx=layer,
                token_positions=token_positions,
                alpha=1,
            ):
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                    model.device
                )
                with torch.no_grad():
                    logits = model(input_ids).logits

                token_ids_at_answer_positions = input_ids[
                    0, answer_indices
                ].tolist()
                values = logits[0, -1, token_ids_at_answer_positions]

                pos_pred = values.argmax().item()

                if generate:
                    pred_ids = model.generate(
                        input_ids,
                        max_new_tokens=schema.max_new_tokens,
                        do_sample=False,
                    )
                    pred = tokenizer.decode(
                        pred_ids[0], skip_special_tokens=True
                    )
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
                results["positional_prediction"].append(
                    metadata["positional"]
                )
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


def save_results(df, fig, output_dir, layer):
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
        description="Run ablation patching experiments on Falcon-H1"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="tiiuae/Falcon-H1-3B-Instruct",
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

    parser.add_argument(
        "--ablate",
        type=str,
        default="none",
        choices=["none", "attention", "mamba"],
        help="Which component to ablate (zero out). "
        "'none': normal experiment. "
        "'attention': zero out attention at ablate-layers. "
        "'mamba': zero out mamba/SSM at ablate-layers.",
    )
    parser.add_argument(
        "--ablate-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to ablate, or 'all' (default: all layers). "
        "Example: '10,11,12,13,14' or 'all'",
    )
    parser.add_argument(
        "--ablate-method",
        type=str,
        default="noise",
        choices=["zero", "noise"],
        help="'zero': zero out the component output. "
        "'noise': add Gaussian noise scaled to noise-scale * output.std()",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=5.0,
        help="Noise multiplier (only used with --ablate-method noise). "
        "noise = randn * output.std() * noise_scale",
    )

    args = parser.parse_args()

    # Parse ablate-layers
    if args.ablate_layers is None or args.ablate_layers == "all":
        ablate_layers = None  # all layers
    else:
        ablate_layers = [int(x.strip()) for x in args.ablate_layers.split(",")]

    schema = get_schema_by_name(args.schema_name)
    print(f"[+] Using schema: {args.schema_name}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name = sanitize_model_name(args.model_id)
        output_dir = Path(f"{model_name}_ablate_{args.ablate}")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"[+] Output directory: {output_dir.absolute()}")

    print(f"[+] Loading model: {args.model_id}")
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }

    if "falcon" in args.model_id.lower() and torch.cuda.is_available():
        try:
            import bitsandbytes

            print(f"[+] Loading in 8-bit for GPU memory efficiency")
            model_kwargs.update(
                {
                    "load_in_8bit": True,
                    "torch_dtype": torch.float16,
                }
            )
        except ImportError:
            print(f"[+] bitsandbytes not available, loading in bfloat16")
    elif not torch.cuda.is_available():
        print(f"[+] No GPU detected, loading in bfloat16 (CPU mode)")
        model_kwargs["device_map"] = None

    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    tokenizer_kwargs = {"token": args.hf_token} if args.hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tokenizer_kwargs)

    num_layers = _num_layers(model)
    print(f"[+] Model has {num_layers} layers")
    print(f"[+] Ablation mode: {args.ablate} ({args.ablate_method}, scale={args.noise_scale})")
    if args.ablate != "none":
        if ablate_layers is None:
            print(f"[+] Ablating {args.ablate} at ALL layers")
        else:
            print(f"[+] Ablating {args.ablate} at layers: {ablate_layers}")

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
    end_layer = (
        args.end_layer if args.end_layer is not None else (num_layers - 1)
    )

    if start_layer < 0:
        raise ValueError(f"start-layer must be >= 0, got {start_layer}")
    if end_layer >= num_layers:
        raise ValueError(
            f"end-layer must be < {num_layers}, got {end_layer}"
        )
    if start_layer > end_layer:
        raise ValueError(
            f"start-layer ({start_layer}) must be <= end-layer ({end_layer})"
        )

    layer_range = range(start_layer, end_layer + 1)

    print(
        f"[+] Running patching for layers {start_layer} to {end_layer} "
        f"with {args.ablate} ablation..."
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
            ablate_mode=args.ablate,
            ablate_layers=ablate_layers,
            ablate_method=args.ablate_method,
            noise_scale=args.noise_scale,
            generate=args.generate,
        )

        fig, ax = plot_patch_effect(
            df, include_reflexive=True, highest_near_pos=0
        )

        save_results(df, fig, output_dir, layer)

        plt.close(fig)

    print(
        f"\n[+] All experiments completed! Results saved in: {output_dir.absolute()}"
    )


if __name__ == "__main__":
    main()
