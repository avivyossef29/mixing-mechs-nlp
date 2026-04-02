#!/usr/bin/env python3
"""
Component activation patching for Falcon-H1.

Instead of patching the full residual stream (Yoav's experiment),
this patches ONLY the mamba output or ONLY the attention output
at a specific layer with the CF's version.

This directly answers: "Which component (mamba or attention) carries
the binding signal at each layer?"

If patching mamba's output at layer 17 → lexical transition appears → mamba writes binding.
If patching attention's output at layer 17 → lexical transition → attention writes binding.
If both show the transition → both carry binding info redundantly.
If neither → binding is in the interaction / residual stream.

Usage:
  # Patch mamba output at layer 17
  python run_falcon_h1_component_patching.py --component mamba --start-layer 17 --end-layer 17

  # Full sweep: both components across layers 0-31
  python run_falcon_h1_component_patching.py --component mamba --num-samples 200

  # Also run the residual stream baseline for comparison
  python run_falcon_h1_component_patching.py --component residual --num-samples 200
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
    run_with_cf_hf,
    patch_component_cf,
    try_schema_checker,
    _num_layers,
)
from plotting import plot_patch_effect


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

    for cur_index in tqdm(range(num_samples), desc=f"Layer {layer} ({component})"):
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

        # Choose patching method
        if component == "residual":
            # Standard residual stream patching (baseline)
            ctx = run_with_cf_hf(
                model, tokenizer, prompt, cf_prompt,
                layer_idx=layer, token_positions=token_positions, alpha=1,
            )
        else:
            # Component-level patching (mamba or attention)
            ctx = patch_component_cf(
                model, tokenizer, prompt, cf_prompt,
                layer_idx=layer, component=component,
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
            results["component"].append(component)

    df = pd.DataFrame(results)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Component activation patching for Falcon-H1"
    )
    parser.add_argument("--model-id", type=str, default="tiiuae/Falcon-H1-3B-Instruct")
    parser.add_argument("--num-instances", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--cat-to-query", type=int, default=1)
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-layer", type=int, default=None)
    parser.add_argument("--end-layer", type=int, default=None)
    parser.add_argument(
        "--component",
        type=str,
        default="mamba",
        choices=["mamba", "attention", "residual"],
        help="Which component to patch with CF's output. "
        "'residual': standard full residual stream patching (baseline). "
        "'mamba': patch only mamba's output. "
        "'attention': patch only attention's output.",
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
        output_dir = Path(f"{model_name}_component_{args.component}")
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"[+] Output directory: {output_dir.absolute()}")

    start_layer = args.start_layer if args.start_layer is not None else 0
    end_layer = args.end_layer if args.end_layer is not None else (num_layers - 1)

    print(f"[+] Running component patching ({args.component}) for layers {start_layer}-{end_layer}")

    for layer in tqdm(range(start_layer, end_layer + 1), desc="Layers"):
        print(f"\n[+] Processing layer {layer}/{num_layers - 1}")

        df = run_experiment_for_layer(
            model, tokenizer, train_ds, schema,
            args.num_instances, args.num_samples,
            layer, args.cat_to_query, args.model_id,
            component=args.component, generate=args.generate,
        )

        # Save results
        layer_dir = output_dir / f"layer_{layer}"
        layer_dir.mkdir(exist_ok=True)

        csv_path = layer_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")

        fig, ax = plot_patch_effect(df, include_reflexive=True, highest_near_pos=0)
        png_path = layer_dir / "patch_effect.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"  Saved PNG: {png_path}")
        plt.close(fig)

        # Print summary for this layer
        counts = df["patch_effect"].value_counts()
        total = len(df)
        summary = {k: f"{counts.get(k, 0)/total*100:.1f}%" for k in
                   ["no_effect", "lexical", "positional", "reflexive", "mixed"]}
        print(f"  Summary: {summary}")

    print(f"\n[+] Done! Results in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
