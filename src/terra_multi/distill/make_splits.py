"""Create the sample-level distillation split."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from terra_multi.distill.data import all_samples, annotated_samples


def partition_samples(samples, annotated, val_frac, seed):
    annotated = set(annotated)
    unannotated = [sample for sample in samples if sample not in annotated]
    if len(unannotated) < 2:
        raise ValueError("at least two unannotated samples are required")
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be between 0 and 1, got {val_frac}")

    order = np.random.default_rng(seed).permutation(len(unannotated))
    n_val = min(max(round(val_frac * len(unannotated)), 1), len(unannotated) - 1)
    validation = {unannotated[i] for i in order[:n_val]}
    return {
        sample: (
            "label_test"
            if sample in annotated
            else "distill_val"
            if sample in validation
            else "distill_train"
        )
        for sample in samples
    }


def build_split_table(data_root, val_frac, seed):
    samples = all_samples(data_root)
    split = partition_samples(samples, annotated_samples(data_root), val_frac, seed)
    return pd.DataFrame(sorted(split.items()), columns=["sample_id", "split"])


def write_split_table(cfg):
    table = build_split_table(cfg["data_root"], cfg["val_frac"], cfg["split_seed"])
    path = Path(cfg["splits_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    path, table = write_split_table(cfg)
    print(f"wrote {path}")
    print(table["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
