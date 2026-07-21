"""Small config-aware LSF submission helper used by repository launchers."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from pathlib import Path

from terra_multi.cohorts import cohort_in_split
from terra_multi.config import config_value, load_config

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
            "if command -v module >/dev/null 2>&1; then module load cellgen/uv; fi\n"
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
        f"#BSUB -n {args.cores}",
        f"#BSUB -q {args.queue}",
    ]
    if args.gpu is not None:
        directives.append(f'#BSUB -gpu "{args.gpu}"')
    directives.extend(
        [
            f"#BSUB -G {args.group}",
            f"#BSUB -M {args.memory}G",
            (
                f'#BSUB -R "select[mem>{args.memory}G] '
                f'rusage[mem={args.memory}G] span[hosts=1]"'
            ),
        ]
    )
    if args.dependency:
        directives.append(f'#BSUB -w "{args.dependency}"')
    command = shlex.join(args.command)
    body = [
        *directives,
        "",
        "set -euo pipefail",
        _environment(args.environment, args.cores),
        f"cd {shlex.quote(str(ROOT))}",
        f"PYTHONPATH=src {command}",
    ]
    return "\n".join(body) + "\n"


def submit(args):
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
    job.add_argument("--queue", required=True)
    job.add_argument("--gpu")
    job.add_argument("--memory", required=True, type=int)
    job.add_argument("--cores", required=True, type=int)
    job.add_argument("--group", default=os.environ.get("BGROUP", "s10396"))
    job.add_argument("--dependency")
    job.add_argument("--environment", choices=("terra", "hest", "none"), default="terra")
    job.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.action == "value":
        print(config_value(args.config, args.key))
    elif args.action == "cohort":
        cohort_path = config_value(args.config, args.cohort_key)
        split_path = config_value(args.config, args.splits_key)
        print(*cohort_in_split(cohort_path, split_path, args.split), sep="\n")
    else:
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            raise ValueError("submit requires a command after --")
        submit(args)


if __name__ == "__main__":
    main()
