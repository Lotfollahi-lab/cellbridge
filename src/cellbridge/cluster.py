"""Submit CellBridge jobs to LSF."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from cellbridge import lsf
from cellbridge.cohorts import cohort_in_split
from cellbridge.config import config_value, load_config

ROOT = Path(__file__).resolve().parents[2]


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _environment(name, cores):
    threads = (
        f"export OMP_NUM_THREADS={cores} OPENBLAS_NUM_THREADS={cores} "
        f"MKL_NUM_THREADS={cores} NUMEXPR_NUM_THREADS={cores}"
    )
    if name == "hest":
        setup = "module load ISG/conda\nconda activate hest"
    elif name == "terra":
        default = ROOT.parents[1] / "envs" / "terra-dev"
        environment = Path(
            os.environ.get("TERRA_ENV", os.environ.get("ENV_PATH", default))
        )
        setup = (
            "module load cellgen/uv\n"
            f"source {shlex.quote(str(environment / 'bin' / 'activate'))}"
        )
    elif name == "none":
        setup = ""
    else:
        raise ValueError(f"unknown environment: {name}")
    return "\n".join(part for part in (setup, threads) if part)


def render_job(args):
    config = load_config(args.config)
    value = config
    for part in args.output_key.split("."):
        value = value[part]
    log_dir = _path(value) / args.log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    directives = [
        f"#BSUB -J {args.job}",
        f"#BSUB -o {log_dir / (args.log_prefix + '_%J.out')}",
        f"#BSUB -e {log_dir / (args.log_prefix + '_%J.err')}",
        f"#BSUB -q {args.queue}",
        f"#BSUB -G {lsf.LSF_GROUP}",
        f"#BSUB -n {args.cores}",
        f"#BSUB -M {args.memory}G",
        (
            f'#BSUB -R "select[mem>{args.memory}G] '
            f'rusage[mem={args.memory}G] span[hosts=1]"'
        ),
    ]
    if args.gmem is not None:
        directives.append(f'#BSUB -gpu "num=1:gmem={args.gmem}"')
    if args.dependency:
        directives.append(f'#BSUB -w "{args.dependency}"')

    command = shlex.join(args.command)
    return (
        "\n".join(
            (
                *directives,
                "",
                "set -euo pipefail",
                _environment(args.environment, args.cores),
                f"cd {shlex.quote(str(ROOT))}",
                f"PYTHONPATH=src {command}",
            )
        )
        + "\n"
    )


def submit(args):
    current = lsf.current_cluster()
    cluster = lsf.resolve_cluster(args.cluster, args.queue, current)
    requested_queue = args.queue
    args.queue = lsf.resolve_queue(cluster, requested_queue, gmem=args.gmem)
    if requested_queue == "auto":
        print(f"using {cluster} queue {args.queue}", file=sys.stderr)

    result = subprocess.run(
        ["bsub"],
        input=render_job(args),
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    match = re.search(r"Job <(\d+)>", result.stdout)
    if not match:
        raise RuntimeError(f"could not parse bsub response: {result.stdout.strip()}")
    print(match.group(1))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    commands.add_parser("queues", help="show known queues and head nodes")

    value = commands.add_parser("value", help="print one expanded config value")
    value.add_argument("config")
    value.add_argument("key")

    cohort = commands.add_parser("cohort", help="print a validated cohort")
    cohort.add_argument("config")
    cohort.add_argument("cohort_key")
    cohort.add_argument("splits_key")
    cohort.add_argument("split")

    job = commands.add_parser("submit", help="submit one LSF job and print its ID")
    job.add_argument("config")
    job.add_argument("--job", required=True)
    job.add_argument("--log-prefix", required=True)
    job.add_argument("--output-key", default="output_dir")
    job.add_argument("--log-subdir", default="logs")
    job.add_argument("--cluster", default=os.environ.get("CELLBRIDGE_CLUSTER", "auto"))
    job.add_argument("--queue", default="auto")
    job.add_argument("--gmem", type=int, help="GPU memory in MB; requests one GPU")
    job.add_argument("--memory", required=True, type=int)
    job.add_argument("--cores", required=True, type=int)
    job.add_argument("--dependency")
    job.add_argument("--environment", choices=("terra", "hest", "none"), default="terra")
    return parser


def _parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = []
    if "--" in argv:
        separator = argv.index("--")
        command = argv[separator + 1 :]
        argv = argv[:separator]
    args = _parser().parse_args(argv)
    args.command = command
    return args


def main(argv=None):
    args = _parse_args(argv)
    if args.action == "queues":
        current = lsf.current_cluster()
        for name in lsf.CLUSTERS:
            marker = "*" if name == current else " "
            print(f"{marker} {lsf.describe_cluster(name)}")
    elif args.action == "value":
        print(config_value(args.config, args.key))
    elif args.action == "cohort":
        cohort_path = config_value(args.config, args.cohort_key)
        split_path = config_value(args.config, args.splits_key)
        print(*cohort_in_split(cohort_path, split_path, args.split), sep="\n")
    else:
        if not args.command:
            raise ValueError("submit requires a command after --")
        if args.gmem is not None and args.gmem <= 0:
            raise ValueError("--gmem must be positive")
        submit(args)


if __name__ == "__main__":
    main()
