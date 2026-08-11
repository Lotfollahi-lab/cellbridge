"""LSF cluster and queue rules for CellBridge jobs."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

CPU_GROUP = "team361"
GPU_GROUP = "s10396"


@dataclass(frozen=True)
class Cluster:
    head: str
    cpu_queues: tuple[str, ...]
    gpu_queues: tuple[str, ...]
    other_queues: tuple[str, ...]

    @property
    def batch_queues(self):
        return self.cpu_queues + self.gpu_queues


CLUSTERS = {
    "cub22": Cluster(
        head="cub22-head1",
        cpu_queues=("oversubscribed",),
        gpu_queues=("inference", "training-normal"),
        other_queues=("system", "cpu-interactive", "gpu-interactive", "transfer"),
    ),
    "tiger22": Cluster(
        head="tiger22-head2",
        cpu_queues=("oversubscribed",),
        gpu_queues=("inference", "training-normal", "training-parallel"),
        other_queues=("system", "transfer"),
    ),
}


def run_command(command) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def current_cluster(run=run_command) -> str:
    output = run(["lsid"])
    match = re.search(r"My cluster name is\s+(\S+)", output)
    if not match or match.group(1) not in CLUSTERS:
        raise RuntimeError("cannot identify the LSF cluster; run 'lsid' on a cluster node")
    return match.group(1)


def resolve_cluster(requested: str, queue: str, current: str) -> str:
    if requested != "auto":
        if requested not in CLUSTERS:
            raise ValueError(f"unknown cluster {requested!r}")
        target = requested
    else:
        candidates = [
            name for name, profile in CLUSTERS.items() if queue in profile.batch_queues
        ]
        target = current if queue == "auto" or current in candidates else None
        if target is None and len(candidates) == 1:
            target = candidates[0]
        if target is None:
            target = current

    if target != current:
        head = CLUSTERS[target].head
        raise RuntimeError(
            f"queue {queue!r} belongs to {target}, but this node uses {current}; "
            f"run 'ssh {head}', return to this checkout, and submit again"
        )
    return target


def resolve_queue(cluster: str, queue: str, *, gmem: int | None) -> str:
    profile = CLUSTERS[cluster]
    if queue == "auto":
        return "training-normal" if gmem is not None else profile.cpu_queues[0]
    if queue in profile.other_queues:
        raise ValueError(f"queue {queue!r} is not a batch-compute queue on {cluster}")
    if queue not in profile.batch_queues:
        choices = ", ".join(profile.batch_queues)
        raise ValueError(
            f"queue {queue!r} is not available on {cluster}; batch queues: {choices}"
        )
    if queue in profile.gpu_queues and gmem is None:
        raise ValueError(f"GPU queue {queue!r} requires --gmem")
    if queue in profile.cpu_queues and gmem is not None:
        raise ValueError(f"CPU queue {queue!r} cannot be used with --gmem")
    return queue


def group_for(gmem: int | None) -> str:
    return GPU_GROUP if gmem is not None else CPU_GROUP


def describe_cluster(name: str) -> str:
    profile = CLUSTERS[name]
    return "\n".join(
        (
            f"{name} ({profile.head})",
            f"  CPU batch: {', '.join(profile.cpu_queues)}",
            f"  GPU batch: {', '.join(profile.gpu_queues)}",
            f"  Other: {', '.join(profile.other_queues)}",
        )
    )
