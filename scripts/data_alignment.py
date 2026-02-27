import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import DEFAULT_MAX_LEN, VOCABULARY


def _validate_sequences(df: pd.DataFrame, max_len: int, vocabulary: str):
    df = df.copy()
    df["length"] = df["sequence"].str.len()
    invalid = df[df["length"] > max_len]
    if not invalid.empty:
        print(f"Warning: {len(invalid)} sequences exceed max_len ({max_len}) and will be clipped.")

    invalid_chars = df[~df["sequence"].apply(lambda seq: set(seq).issubset(set(vocabulary)))]
    if not invalid_chars.empty:
        print(f"Warning: {len(invalid_chars)} sequences contain non-standard amino acids.")

    return df


def _make_splits(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    temp_frac = val_frac + test_frac
    split_df = df.copy()
    stratify_col = split_df["label"]
    train_df, temp_df = train_test_split(
        split_df,
        test_size=temp_frac,
        stratify=stratify_col,
        random_state=seed,
    )

    if val_frac == 0 and test_frac == 0:
        train_df = train_df.assign(split="train")
        return train_df

    relative_val = val_frac / temp_frac if temp_frac > 0 else 0
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac / temp_frac if temp_frac else 0,
        stratify=temp_df["label"],
        random_state=seed + 1,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def _collect_stats(df: pd.DataFrame):
    df = df.copy()
    df["length"] = df["sequence"].str.len()
    length_summary = df["length"].describe().to_dict()
    label_counts = df["label"].value_counts().to_dict()
    aa_counter = Counter("".join(df["sequence"].tolist()))
    vocab_freq = {aa: aa_counter.get(aa, 0) for aa in VOCABULARY}
    return {
        "length_summary": length_summary,
        "label_counts": label_counts,
        "amino_acid_counts": vocab_freq,
        "total_sequences": len(df),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Align AMP dataset (train/val/test split + shared metadata)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("preprocessed_data/classify.csv"),
        help="Featurized classify CSV (sequence + label).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preprocessed_data/classify_splits.csv"),
        help="Path to write the split CSV with an added `split` column.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("preprocessed_data/data_alignment_summary.json"),
        help="Path to persist computed statistics for downstream validation.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1,
        help="Fraction of data allocated to the validation split.",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Fraction of data allocated to the test split.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help="Enforce maximum sequence length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible split assignment.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Expected {args.input} to exist before aligning data.")

    df = pd.read_csv(args.input)
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain `sequence` and `label` columns.")

    df = _validate_sequences(df, args.max_len, VOCABULARY)
    split_df = _make_splits(df, args.val_frac, args.test_frac, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output, index=False)
    print(f"Wrote aligned dataset with split column to {args.output}")

    stats = _collect_stats(split_df)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(stats, indent=2))
    print(f"Wrote dataset summary to {args.metadata}")
    print(f"Length summary: {stats['length_summary']}")
    print(f"Label distribution: {stats['label_counts']}")


if __name__ == "__main__":
    main()
