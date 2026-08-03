"""Live LSF capacity checks so submissions fail fast instead of pending forever.

Two mistakes cost the most time on this cluster. Asking a queue for more memory
than its hosts currently advertise pends forever behind an opaque "There are no
suitable hosts for the job", and sending a CPU-only job to a GPU-backed queue is
rejected by esub with a message that never names the queue. Both are cheap to
detect before bsub, so this module reads live capacity and turns either one into
an actionable error.

Every check fails open: if a command is missing or its output cannot be parsed,
the caller is told nothing is known rather than being blocked.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

# Queues this repository is allowed to route work to, most-preferred first.
# gpu-interactive reaches idle hardware but is reserved for interactive
# sessions, and transfer/system are not general compute, so none are listed.
CPU_QUEUES = tuple(
    part for part in os.environ.get("TERRA_CPU_QUEUES", "oversubscribed").split(",")
    if part
)
GPU_QUEUES = tuple(
    part
    for part in os.environ.get(
        "TERRA_GPU_QUEUES", "training-normal,inference"
    ).split(",")
    if part
)

_MEMORY_SCALE = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}


def run_command(command) -> str:
    """Return stdout, or an empty string if the command is absent or fails."""
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def parse_memory_mb(text):
    """Convert an LSF size such as ``7.7G`` or ``512M`` to megabytes."""
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([KMGTkmgt])?\s*", str(text))
    if not match:
        return None
    value = float(match.group(1))
    return value * _MEMORY_SCALE[(match.group(2) or "M").upper()]


def queue_host_groups(output):
    """Extract host-group names from ``bqueues -l`` (``HOSTS: name/ other/``)."""
    for line in output.splitlines():
        if line.strip().upper().startswith("HOSTS:"):
            entries = line.split(":", 1)[1].split()
            return [entry.rstrip("/") for entry in entries if entry not in {"", "all"}]
    return []


def group_hosts(output):
    """Extract member host names from ``bmgroup <group>``."""
    hosts = []
    for line in output.splitlines()[1:]:
        hosts.extend(line.split()[1:])
    return hosts


def free_memory_mb(output):
    """Largest currently free memory across the ``lsload -w`` rows, in MB."""
    best = None
    for line in output.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 12 or "ok" not in fields[1]:
            continue
        memory = parse_memory_mb(fields[-1])
        if memory is not None and (best is None or memory > best):
            best = memory
    return best


def free_slots(output):
    """Total unused job slots across the ``bhosts -w`` rows."""
    total = 0
    for line in output.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 6 or fields[1] != "ok":
            continue
        try:
            total += max(0, int(fields[3]) - int(fields[4]))
        except ValueError:
            continue
    return total


def free_gpus(output):
    """Count GPUs with no running job in ``bhosts -gpu`` output."""
    count = 0
    for line in output.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 4 and fields[-4:-3] and fields[-4].isdigit():
            count += int(fields[-4]) == 0
    return count


def has_gpus(output):
    """True when ``bhosts -gpu`` returned any GPU row for the hosts asked about."""
    return any(
        len(line.split()) >= 9 and "NVIDIA" in line.upper()
        for line in output.splitlines()[1:]
    )


@dataclass(frozen=True)
class Capacity:
    queue: str
    hosts: tuple
    max_free_memory_mb: float | None
    free_slots: int
    gpu_backed: bool
    free_gpus: int

    @property
    def known(self):
        return bool(self.hosts) and self.max_free_memory_mb is not None


def queue_capacity(queue, run=run_command) -> Capacity:
    """Read the live host set and free capacity behind one queue."""
    groups = queue_host_groups(run(["bqueues", "-l", str(queue)]))
    hosts = []
    for group in groups:
        hosts.extend(group_hosts(run(["bmgroup", group])))
    hosts = tuple(dict.fromkeys(hosts))
    if not hosts:
        return Capacity(str(queue), (), None, 0, False, 0)
    gpu_output = run(["bhosts", "-gpu", *hosts])
    return Capacity(
        queue=str(queue),
        hosts=hosts,
        max_free_memory_mb=free_memory_mb(run(["lsload", "-w", *hosts])),
        free_slots=free_slots(run(["bhosts", "-w", *hosts])),
        gpu_backed=has_gpus(gpu_output),
        free_gpus=free_gpus(gpu_output),
    )


def request_problems(capacity: Capacity, *, memory_gb, cores, gpu):
    """Return blocking reasons why this request cannot run on this queue."""
    problems = []
    if capacity.gpu_backed and gpu is None:
        alternative = next(iter(CPU_QUEUES), None)
        problems.append(
            f"queue {capacity.queue!r} is GPU-backed, so LSF rejects a job with no "
            f"--gpu selector. Pass --gpu - to reserve a GPU anyway, or send "
            f"CPU-only work to {alternative!r}."
        )
    # Nothing is reported for the reverse case (a GPU request on a CPU queue):
    # bhosts -gpu also returns no rows while mbatchd is still refreshing its GPU
    # view, so treating "no rows" as "no GPUs" would block valid submissions.
    if capacity.known and memory_gb * 1024.0 > capacity.max_free_memory_mb:
        problems.append(
            f"queue {capacity.queue!r} advertises at most "
            f"{capacity.max_free_memory_mb / 1024.0:.1f}G free on any host, "
            f"below the {memory_gb}G requested; the job would pend indefinitely. "
            f"Lower --memory or choose a larger queue."
        )
    return problems


def choose_queue(*, memory_gb, cores, gpu, run=run_command):
    """Pick the preferred queue that fits and can start the job now.

    Falls back to a queue that merely fits, then to the first candidate, so a
    cluster this cannot introspect still gets a submission rather than an error.
    Anything genuinely unrunnable is caught by the caller's preflight instead.
    """
    # CPU work prefers the CPU queues but may spill onto a GPU queue, because
    # the CPU hosts here cap out near 8G and jobs like the quality check ask for
    # 64G. Reaching those hosts costs a reserved GPU, so it is strictly a
    # fallback and never the first choice.
    wanted = (
        GPU_QUEUES if gpu not in (None, "-") else tuple(CPU_QUEUES) + tuple(GPU_QUEUES)
    )
    if not wanted:
        return None
    fits = None
    for queue in wanted:
        capacity = queue_capacity(queue, run=run)
        if not capacity.known or request_problems(
            capacity, memory_gb=memory_gb, cores=cores, gpu=gpu or "-"
        ):
            continue
        if capacity.free_slots >= cores:
            return queue
        fits = fits or queue
    return fits or wanted[0]
