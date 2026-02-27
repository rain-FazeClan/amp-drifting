import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from experience_filter import ExperienceFilter
from scripts.pipeline_core import (
    ConditioningSpec,
    PipelineStats,
    build_heuristics_map,
    classify_sequences,
    evaluate_heuristics,
    instantiate_generator,
    load_predictive_model,
    parse_device,
)
from utils import DEFAULT_MAX_LEN


def run_pipeline(args):
    device = parse_device(args.device)
    model_path = Path(args.diffusion_model if args.generator == "diffusion" else args.jst_model)
    generator = instantiate_generator(args.generator, model_path, args.max_length, device)
    classifier = load_predictive_model(Path(args.classifier))
    filter_obj = ExperienceFilter()
    spec = ConditioningSpec(
        min_length=args.min_length,
        max_length=args.max_length,
        min_charge=args.min_charge,
        min_amphipathicity=args.min_amphipathy,
        ph=args.ph,
    )

    stats = PipelineStats()
    candidates = []
    start = time.time()

    while len(candidates) < args.target_candidates:
        if stats.batches >= args.max_batches or stats.total_generated >= args.max_generated:
            print("Reached generation limit before hitting target candidates.")
            break

        batch = generator.generate_batch(args.batch_size, args.temperature)
        if not batch:
            print("Generator returned no sequences; stopping.")
            break

        stats.batches += 1
        stats.total_generated += len(batch)

        heuristics_passed = []
        for seq in batch:
            reasons, charge, amph = evaluate_heuristics(seq, filter_obj, spec)
            if reasons:
                for reason in reasons:
                    stats.heuristic_failures[reason] += 1
                continue
            stats.heuristic_passed += 1
            heuristics_passed.append(
                {"sequence": seq, "charge": charge, "amphipathicity": amph}
            )

        heuristics_map = build_heuristics_map(heuristics_passed)
        sequence_list = [info["sequence"] for info in heuristics_passed]
        batch_candidates = classify_sequences(
            sequence_list, classifier, args.amp_threshold, heuristics_map, stats
        )
        for candidate in batch_candidates:
            candidate["generator"] = args.generator
            candidate["batch_index"] = stats.batches
            candidate["timestamp"] = time.time()
            candidates.append(candidate)

    stats.duration_sec = time.time() - start
    return candidates, stats


def save_results(candidates: list, stats: PipelineStats, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    candidate_path = output_dir / f"pipeline_{args.generator}_{ts}.csv"
    summary_path = output_dir / f"pipeline_{args.generator}_{ts}.json"

    if candidates:
        pd.DataFrame(candidates).to_csv(candidate_path, index=False)
        print(f"Saved {len(candidates)} candidates to {candidate_path}")
    else:
        print("No candidates reached the threshold.")

    summary = {
        "generator": args.generator,
        "target_candidates": args.target_candidates,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "amp_threshold": args.amp_threshold,
        "max_batches": args.max_batches,
        "max_generated": args.max_generated,
        "stats": {
            "total_generated": stats.total_generated,
            "batches": stats.batches,
            "heuristic_passed": stats.heuristic_passed,
            "classification_checked": stats.classification_checked,
            "total_candidates": stats.total_candidates,
            "duration_sec": stats.duration_sec,
            "heuristic_failures": dict(stats.heuristic_failures),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare generator + filter pipeline efficiency."
    )
    parser.add_argument(
        "--generator",
        choices=["diffusion", "jst"],
        default="diffusion",
        help="Which generator to evaluate.",
    )
    parser.add_argument(
        "--diffusion_model",
        default="models/diffusion_model_transformer.pth",
        help="Diffusion model path for the legacy generator.",
    )
    parser.add_argument(
        "--jst_model",
        default="models/diffusion_model_x_pred.pth",
        help="JsT generator checkpoint.",
    )
    parser.add_argument("--classifier", default="models/predictive_model.pkl")
    parser.add_argument("--target_candidates", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--amp_threshold", type=float, default=0.8)
    parser.add_argument("--min_length", type=int, default=6)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--min_charge", type=float, default=3.5)
    parser.add_argument("--min_amphipathy", type=float, default=0.35)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--max_generated", type=int, default=10000)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device to use (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--output_dir",
        default="results/pipeline_runs",
        help="Directory to persist pipeline outputs.",
    )

    args = parser.parse_args()

    candidates, stats = run_pipeline(args)
    save_results(candidates, stats, args)


if __name__ == "__main__":
    main()
