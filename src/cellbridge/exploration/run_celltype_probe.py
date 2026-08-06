"""Compatibility entry point for the historical HEST linear probe."""

import argparse


from cellbridge.config import load_config

from cellbridge.evaluation.hest import MODELS, load_labels, run_hest_evaluation
from cellbridge.evaluation.linear_probe import reduce_features


def apply_uni_pca(uni_embeddings, train_idx, n_components=256):
    """Compatibility wrapper around the shared train-only PCA."""
    return reduce_features(uni_embeddings, train_idx, n_components, seed=0)


def main():
    parser = argparse.ArgumentParser(description="Run one cell-type linear probe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_hest_evaluation(cfg, args.model, "linear")


if __name__ == "__main__":
    main()


__all__ = ["MODELS", "apply_uni_pca", "load_labels", "main"]
