"""Run linear and label-reference kNN evaluation from one HEST config."""

import argparse

import yaml

from terra_multi.evaluation.classification import METHODS, resolve_methods
from terra_multi.evaluation.hest import MODELS, run_hest_evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--method", action="append", choices=METHODS)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    models = args.model or cfg.get("models", ["terra", "uni", "concat"])
    methods = resolve_methods(cfg, args.method)

    for model in models:
        run_hest_evaluations(cfg, model, methods)


if __name__ == "__main__":
    main()
