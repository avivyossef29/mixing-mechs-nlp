#!/usr/bin/env python3
"""
Baseline accuracy test for Falcon-H1 with component ablation.

Tests whether the model can still solve the binding task when
one component (attention or mamba) is disrupted via zeroing or noise.

Usage:
  # Zero out mamba at all layers
  python run_falcon_h1_ablation_baseline.py --ablate mamba --ablate-method zero

  # Add noise (5x std) to attention at layers 17-26
  python run_falcon_h1_ablation_baseline.py --ablate attention --ablate-method noise --noise-scale 5 --ablate-layers 17,18,19,20,21,22,23,24,25,26

  # Sweep: test multiple noise scales and both components
  python run_falcon_h1_ablation_baseline.py --sweep --num-samples 100
"""

import os
import sys
import argparse
import logging
import random
import csv

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append("CausalAbstraction")

logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
import transformers

transformers.logging.set_verbosity_error()

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

from grammar.schemas import SCHEMA_BOXES
from grammar.task_to_causal_model import (
    multi_order_multi_schema_task_to_lookbacks_generic_causal_model,
)
from training import (
    sample_answerable_question_template,
    get_counterfactual_datasets,
    ppkn_simpler_counterfactual_template_split_key_loc,
)
from tasks.dist import (
    get_end_str,
    format_prompt,
    to_str_tokens,
    try_schema_checker,
    _num_layers,
)


def install_hooks(model, ablate_target, ablate_layers, method="zero", noise_scale=1.0):
    """Install ablation hooks on attention or mamba at specified layers.

    method='zero': zero out the output
    method='noise': add Gaussian noise scaled to noise_scale * output.std()
    """
    handles = []
    num_layers = len(model.model.layers)

    if ablate_layers is None:
        ablate_layers = list(range(num_layers))

    for layer_idx in ablate_layers:
        layer = model.model.layers[layer_idx]

        if ablate_target == "attention":
            target = layer.self_attn
        elif ablate_target == "mamba":
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

        handles.append(target.register_forward_hook(make_hook()))

    return handles


def run_accuracy_test(model, tokenizer, train, schema, num_samples):
    """Run the binding task and return (correct, total)."""
    correct = 0
    total = 0

    model.eval()
    for i in tqdm(range(num_samples), desc="Testing"):
        prompt = format_prompt(tokenizer, train[i]["input"]["raw_input"])
        metadata = train[i]["input"]["metadata"]
        expected = metadata["no_effect"]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            logits = model(input_ids).logits

        prompt_str_tokenized = to_str_tokens(tokenizer, prompt)
        answer_indices = []
        for j, token in enumerate(prompt_str_tokenized):
            if schema.matchers[1](token):
                answer_indices.append(j)

        token_ids_at_answer_positions = input_ids[0, answer_indices].tolist()
        values = logits[0, -1, token_ids_at_answer_positions]
        pred_idx = values.argmax().item()
        pred = prompt_str_tokenized[answer_indices[pred_idx]]
        pred = re.sub(r"\s+", "", pred.strip())

        is_correct = try_schema_checker(pred, expected, schema)
        if is_correct:
            correct += 1
        total += 1

    return correct, total


def main():
    parser = argparse.ArgumentParser(
        description="Baseline accuracy test: can Falcon-H1 solve binding with one component ablated?"
    )
    parser.add_argument("--model-id", type=str, default="tiiuae/Falcon-H1-3B-Instruct")
    parser.add_argument("--num-instances", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--ablate",
        type=str,
        default="none",
        choices=["none", "attention", "mamba"],
    )
    parser.add_argument(
        "--ablate-method",
        type=str,
        default="noise",
        choices=["zero", "noise"],
    )
    parser.add_argument("--noise-scale", type=float, default=5.0)
    parser.add_argument(
        "--ablate-layers",
        type=str,
        default=None,
        help="Comma-separated layer indices or 'all' (default: all)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run full sweep: both targets x multiple noise scales x layer ranges",
    )

    args = parser.parse_args()

    # Parse ablate-layers
    if args.ablate_layers is None or args.ablate_layers == "all":
        ablate_layers = None
    else:
        ablate_layers = [int(x.strip()) for x in args.ablate_layers.split(",")]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[+] Loading model: {args.model_id}")
    model_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if not torch.cuda.is_available():
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    num_layers = _num_layers(model)
    print(f"[+] Model has {num_layers} layers")

    # Generate dataset
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

    train = train_ds[schema.name][schema.name]

    # Output directory
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.sweep:
        # Full sweep mode
        noise_scales = [1, 3, 5, 7, 10]
        targets = ["mamba", "attention"]
        layer_ranges = {
            "all": None,
            "0-16": list(range(0, 17)),
            "17-21": list(range(17, 22)),
            "22-26": list(range(22, 27)),
            "27-31": list(range(27, 32)),
        }

        results = []

        # First: baseline (no ablation)
        print(f"\n{'='*60}")
        print(f"Baseline (no ablation)")
        print(f"{'='*60}")
        correct, total = run_accuracy_test(model, tokenizer, train, schema, args.num_samples)
        acc = correct / total * 100
        print(f"  Accuracy: {acc:.1f}% ({correct}/{total})")
        results.append({
            "target": "none", "method": "none", "noise_scale": 0,
            "layers": "none", "correct": correct, "total": total, "accuracy": acc,
        })

        # Sweep noise
        for target in targets:
            for scale in noise_scales:
                for layer_desc, layers in layer_ranges.items():
                    print(f"\n{'='*60}")
                    print(f"Noise on {target} @ layers {layer_desc}, scale={scale}")
                    print(f"{'='*60}")

                    handles = install_hooks(model, target, layers, "noise", scale)
                    correct, total = run_accuracy_test(model, tokenizer, train, schema, args.num_samples)
                    for h in handles:
                        h.remove()

                    acc = correct / total * 100
                    print(f"  Accuracy: {acc:.1f}% ({correct}/{total})")
                    results.append({
                        "target": target, "method": "noise", "noise_scale": scale,
                        "layers": layer_desc, "correct": correct, "total": total, "accuracy": acc,
                    })

        # Save results
        if output_dir:
            csv_path = os.path.join(output_dir, "ablation_sweep_results.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\n[+] Results saved to {csv_path}")

        # Print summary table
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"{'Target':<12} {'Scale':<8} {'Layers':<10} {'Accuracy':<15}")
        print(f"{'-'*45}")
        for r in results:
            print(f"{r['target']:<12} {r['noise_scale']:<8} {r['layers']:<10} {r['accuracy']:.1f}% ({r['correct']}/{r['total']})")

    else:
        # Single run mode
        print(f"[+] Ablation: {args.ablate} ({args.ablate_method}, scale={args.noise_scale})")
        if args.ablate != "none":
            desc = "ALL" if ablate_layers is None else str(ablate_layers)
            print(f"[+] Ablating at layers: {desc}")

        handles = []
        if args.ablate != "none":
            handles = install_hooks(
                model, args.ablate, ablate_layers, args.ablate_method, args.noise_scale
            )
            print(f"[+] Installed {len(handles)} hooks")

        correct, total = run_accuracy_test(model, tokenizer, train, schema, args.num_samples)

        for h in handles:
            h.remove()

        accuracy = correct / total * 100
        print(f"\n{'='*50}")
        print(f"Model: {args.model_id}")
        print(f"Ablation: {args.ablate} {args.ablate_method} (scale={args.noise_scale})")
        print(f"Layers: {'all' if ablate_layers is None else ablate_layers}")
        print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
        print(f"{'='*50}")

        if output_dir:
            csv_path = os.path.join(output_dir, "ablation_baseline_results.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "target", "method", "noise_scale", "layers", "correct", "total", "accuracy"
                ])
                writer.writeheader()
                writer.writerow({
                    "target": args.ablate, "method": args.ablate_method,
                    "noise_scale": args.noise_scale,
                    "layers": "all" if ablate_layers is None else str(ablate_layers),
                    "correct": correct, "total": total, "accuracy": accuracy,
                })
            print(f"[+] Results saved to {csv_path}")


if __name__ == "__main__":
    main()
