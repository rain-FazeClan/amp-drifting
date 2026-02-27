import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from experience_filter import ExperienceFilter
from generate_peptides import calculate_features_for_sequences
from scripts.pipeline_core import (
    ConditioningSpec,
    instantiate_generator,
    load_predictive_model,
    parse_device,
)


def evaluate_conditioning(
    generator_name: str,
    model_path: Path,
    spec: ConditioningSpec,
    args,
    device: torch.device,
):
    generator = instantiate_generator(generator_name, model_path, spec.max_length, device)
    sequences = generator.generate_sequences(args.sample_size, args.batch_size, args.temperature)

    filter_obj = ExperienceFilter()
    inspected = []
    for seq in sequences:
        reasons, charge, amph = evaluate_formula(seq, filter_obj, spec)
        inspected.append(
            {
                "sequence": seq,
                "failures": reasons,
                "net_charge": charge,
                "amphipathicity": amph,
            }
        )

    hits = [entry for entry in inspected if not entry["failures"]]
    failure_counter = Counter(reason for entry in inspected for reason in entry["failures"])

    candidate_summary = {
        "generator": generator_name,
        "total_samples": len(sequences),
        "hits": len(hits),
        "hit_rate": len(hits) / len(sequences) if sequences else 0,
        "failure_modes": dict(failure_counter),
        "avg_length_hits": np.mean([len(entry["sequence"]) for entry in hits]) if hits else 0,
        "avg_charge_hits": np.mean([entry["net_charge"] for entry in hits]) if hits else 0,
        "avg_amphipathicity_hits": np.mean(
            [entry["amphipathicity"] for entry in hits]
        )
        if hits
        else 0,
    }

    if args.classifier and hits:
        classifier = load_predictive_model(Path(args.classifier))
        hit_sequences = [entry["sequence"] for entry in hits]
        features, sequences_series = calculate_features_for_sequences(hit_sequences)
        if not features.empty:
            probs = classifier.predict_proba(features)[:, 1]
            candidate_summary["average_amp_probability"] = float(np.mean(probs))
            candidate_summary["amp_hit_rate"] = float(
                np.mean(probs >= args.amp_threshold)
            )
        else:
            candidate_summary["average_amp_probability"] = None
            candidate_summary["amp_hit_rate"] = 0.0
    else:
        candidate_summary["average_amp_probability"] = None
        candidate_summary["amp_hit_rate"] = 0.0

    return candidate_summary, inspected


def evaluate_formula(sequence, filter_obj, spec):
    reasons = []
    if not filter_obj.check_length(sequence, spec.min_length, spec.max_length):
        reasons.append("length")
    charge = filter_obj.calculate_net_charge_at_ph(sequence, spec.ph)
    if charge < spec.min_charge:
        reasons.append("charge")
    amphipathicity = filter_obj.calculate_amphipathicity(sequence)
    if amphipathicity < spec.min_amphipathicity:
        reasons.append("amphipathicity")
    return reasons, charge, amphipathicity


def main():
    parser = argparse.ArgumentParser(description="Conditioning benchmark for generators.")
    parser.add_argument(
        "--generators",
        nargs="+",
        choices=["diffusion", "jst"],
        default=["diffusion", "jst"],
        help="Generator(s) to evaluate.",
    )
    parser.add_argument(
        "--diffusion_model",
        default="models/diffusion_model_transformer.pth",
        help="Path to the diffusion generator checkpoint.",
    )
    parser.add_argument(
        "--jst_model",
        default="models/diffusion_model_x_pred.pth",
        help="Path to the JsT generator checkpoint.",
    )
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--amp_threshold", type=float, default=0.8)
    parser.add_argument("--min_length", type=int, default=15)
    parser.add_argument("--max_length", type=int, default=15)
    parser.add_argument("--min_charge", type=float, default=4.0)
    parser.add_argument("--min_amphipathy", type=float, default=0.45)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--classifier", default="models/predictive_model.pkl")
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for generation (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--output_dir",
        default="results/conditioning",
        help="Directory to write conditioning summaries.",
    )

    args = parser.parse_args()
    device = parse_device(args.device)
    spec = ConditioningSpec(
        min_length=args.min_length,
        max_length=args.max_length,
        min_charge=args.min_charge,
        min_amphipathicity=args.min_amphipathy,
        ph=args.ph,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_summary = []

    for generator_name in args.generators:
        model_path = (
            Path(args.diffusion_model)
            if generator_name == "diffusion"
            else Path(args.jst_model)
        )
        start = time.time()
        summary, inspected = evaluate_conditioning(
            generator_name, model_path, spec, args, device
        )
        summary["runtime_sec"] = time.time() - start
        overall_summary.append(summary)

        csv_path = output_dir / f"conditioning_{generator_name}_{int(time.time())}.csv"
        pd.DataFrame(inspected).to_csv(csv_path, index=False)
        print(f"Wrote detailed sequences for {generator_name} to {csv_path}")

    summary_path = output_dir / f"conditioning_summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(overall_summary, indent=2))
    print(f"Conditioning summary saved to {summary_path}")


if __name__ == "__main__":
    main()
