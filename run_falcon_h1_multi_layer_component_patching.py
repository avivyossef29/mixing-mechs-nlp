#!/usr/bin/env python3
"""
Multi-layer component activation patching for Falcon-H1.

Patches a component (mamba or attention) at ALL layers simultaneously
with the CF's version, then classifies the result.

This answers: "Does the cumulative contribution of mamba (or attention)
across the entire network carry the binding signal?"

The single-layer experiment showed 100% no_effect for both mamba and
attention — likely because each component's per-layer contribution is
tiny (mamba × 0.088, attention × 0.15). This experiment tests whether
the signal is there but distributed across layers.

Usage:
  # Patch mamba at all 32 layers simultaneously
  python run_falcon_h1_multi_layer_component_patching.py --component mamba

  # Patch attention at all layers
  python run_falcon_h1_multi_layer_component_patching.py --component attention
"""

import os
import sys
import argparse
import logging
import random
from pathlib import Path

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
from grammar.schemas import SCHEMA_BOXES
from tasks.dist import (
    get_end_str,
    format_prompt,
    to_str_tokens,
    patch_component_cf_multi_layer,
    try_schema_checker,
    _num_layers,
)
from plotting import plot_patch_effect


def run_experiment_multi_layer(
    model,
    tokenizer,
    train_ds,
    schema,
    num_instances,
    num_samples,
    layer_indices,
    cat_to_query,
    model_id,
    component="mamba",
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
        "component": [],
    }

    train = train_ds[schema.name][schema.name]
    token_positions = [-1]
    end_str = get_end_str(model_id)
    model_id_str = model_id
    layers_str = f"{layer_indices[0]}-{layer_indices[-1]}"

    for cur_index in tqdm(range(num_samples), desc=f"Layers {layers_str} ({component})"):
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

        assert len(answer_indices) == num_instances
        assert keyload_index is not None
        assert payload_index is not None

        pos_index = metadata["dst_index"]

        ctx = patch_component_cf_multi_layer(
            model, tokenizer, prompt, cf_prompt,
            layer_indices=layer_indices, component=component,
            token_positions=token_positions, alpha=1,
        )

        with ctx:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )
            with torch.no_grad():
                logits = model(input_ids).logits

            token_ids_at_answer_positions = input_ids[0, answer_indices].tolist()
            values = logits[0, -1, token_ids_at_answer_positions]

            pos_pred = values.argmax().item()

            if generate:
                pred_ids = model.generate(
                    input_ids, max_new_tokens=schema.max_new_tokens, do_sample=False,
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
            results["layer"].append(layers_str)
            results["positional_prediction"].append(metadata["positional"])
            results["payload_prediction"].append(metadata["payload"])
            results["keyload_prediction"].append(metadata["keyload"])
            results["no_effect_prediction"].append(metadata["no_effect"])
            results["patch_effect"].append(patch_effect)
            results["prediction"].append(pred)
            results["dist"].append(values.tolist())
            results["distance"].append(pos_index - pos_pred)
            results["generated"].append(generate)
            results["component"].append(component)

    df = pd.DataFrame(results)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Multi-layer component activation patching for Falcon-H1"
    )
    parser.add_argument("--model-id", type=str, default="tiiuae/Falcon-H1-3B-Instruct")
    parser.add_argument("--num-instances", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--cat-to-query", type=int, default=1)
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-layer", type=int, default=None,
                        help="First layer to include in multi-layer patch (default: 0)")
    parser.add_argument("--end-layer", type=int, default=None,
                        help="Last layer to include in multi-layer patch (default: last)")
    parser.add_argument(
        "--component",
        type=str,
        default="mamba",
        choices=["mamba", "attention"],
        help="Which component to patch at all layers.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[+] Loading model: {args.model_id}")
    model_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}

    if "falcon" in args.model_id.lower() and torch.cuda.is_available():
        try:
            import bitsandbytes
            print(f"[+] Loading in 8-bit for GPU memory efficiency")
            model_kwargs.update({"load_in_8bit": True, "torch_dtype": torch.float16})
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
    print(f"[+] Component: {args.component}")

    schema = SCHEMA_BOXES
    causal_model = multi_order_multi_schema_task_to_lookbacks_generic_causal_model(
        [schema], args.num_instances, num_fillers_per_item=0, fillers=False
    )
    causal_models = {schema.name: causal_model}

    train_ds, _, _ = get_counterfactual_datasets(
        None,
        [schema],
        num_samples=args.num_samples,
        num_instances=args.num_instances,
        cat_indices_to_query=[0],
        answer_cat_id=1,
        do_assert=True,
        do_filter=False,
        counterfactual_template=ppkn_simpler_counterfactual_template_split_key_loc,
        causal_models=causal_models,
        sample_an_answerable_question=sample_answerable_question_template,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_name = args.model_id.replace("/", "_")
        output_dir = Path(f"{model_name}_multi_layer_{args.component}")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"[+] Output directory: {output_dir.absolute()}")

    start_layer = args.start_layer if args.start_layer is not None else 0
    end_layer = args.end_layer if args.end_layer is not None else (num_layers - 1)
    layer_indices = list(range(start_layer, end_layer + 1))

    print(f"[+] Patching {args.component} at ALL layers {start_layer}-{end_layer} simultaneously")

    df = run_experiment_multi_layer(
        model, tokenizer, train_ds, schema,
        args.num_instances, args.num_samples,
        layer_indices, args.cat_to_query, args.model_id,
        component=args.component, generate=args.generate,
    )

    # Save results
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    fig, ax = plot_patch_effect(df, include_reflexive=True, highest_near_pos=0)
    png_path = output_dir / "patch_effect.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  Saved PNG: {png_path}")
    plt.close(fig)

    # Print summary
    counts = df["patch_effect"].value_counts()
    total = len(df)
    summary = {k: f"{counts.get(k, 0)/total*100:.1f}%" for k in
               ["no_effect", "lexical", "positional", "reflexive", "mixed"]}
    print(f"\n[+] Summary ({args.component} patched at layers {start_layer}-{end_layer}):")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    print(f"\n[+] Done! Results in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
