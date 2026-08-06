import pytest

from cellbridge import lsf

# Captured verbatim from the cluster this repository runs on.
BQUEUES_OVERSUBSCRIBED = """
QUEUE: oversubscribed
  -- For running oversubscribed applications
USERS: all
HOSTS:  oversubscribed_hosts/
RES_REQ:  "select[type==any] span[hosts=1]"
"""
BQUEUES_MULTI_GROUP = "HOSTS:  standard_hosts/ oversubscribed_hosts/ \n"
BQUEUES_ALL_HOSTS = "HOSTS:  all\n"

BMGROUP = (
    "GROUP_NAME    HOSTS                     \n"
    "oversubscrib cub22-oversubscribed03 cub22-oversubscribed01 "
    "cub22-oversubscribed02 \n"
)
LSLOAD = (
    "HOST_NAME               status  r15s   r1m  r15m   ut    pg  ls    it   "
    "tmp   swp   mem\n"
    "cub22-oversubscribed01      ok   0.0   0.0   1.6   0%   0.0   0   367   "
    "17G 15.7G  7.7G\n"
    "cub22-oversubscribed02      ok   0.0   0.0   0.0   0%   0.0   0   147   "
    "17G 15.9G    8G\n"
)
LSLOAD_BIG = (
    "HOST_NAME               status  r15s   r1m  r15m   ut    pg  ls    it   "
    "tmp   swp   mem\n"
    "farm-gpu0301                ok   7.3   8.3   7.7   3%   0.0   0  3422 "
    "1377G  182G  1.7T\n"
)
BHOSTS = (
    "HOST_NAME          STATUS       JL/U    MAX  NJOBS    RUN  SSUSP  USUSP    RSV \n"
    "cub22-oversubscribed01 ok           -     30      0      0      0      0      0\n"
    "cub22-oversubscribed03 closed       -     30      0      0      0      0      0\n"
)
BHOSTS_GPU = (
    "HOST_NAME            GPU_ID           MODEL     MUSED      MRSV  NJOBS    "
    "RUN   SUSP    RSV \n"
    "farm-gpu0301              0 NVIDIAH10080GBH      564M        0M      0      "
    "0      0      0\n"
    "                          1 NVIDIAH10080GBH     17.2G        0M      1      "
    "1      0      0\n"
)
BHOSTS_GPU_NONE = (
    "bhosts: Waiting for mbatchd to update host GPU information from LIM or the "
    "hosts that you specified do not have GPUs.\n"
)


@pytest.mark.parametrize(
    "text,expected",
    [("7.7G", 7884.8), ("8G", 8192.0), ("512M", 512.0), ("1.7T", 1782579.2)],
)
def test_parse_memory_accepts_lsf_size_suffixes(text, expected):
    assert lsf.parse_memory_mb(text) == pytest.approx(expected)


def test_parse_memory_rejects_unparseable_values():
    assert lsf.parse_memory_mb("-") is None


def test_queue_host_groups_strips_markers_and_ignores_all():
    assert lsf.queue_host_groups(BQUEUES_OVERSUBSCRIBED) == ["oversubscribed_hosts"]
    assert lsf.queue_host_groups(BQUEUES_MULTI_GROUP) == [
        "standard_hosts",
        "oversubscribed_hosts",
    ]
    assert lsf.queue_host_groups(BQUEUES_ALL_HOSTS) == []


def test_group_hosts_reads_every_member():
    assert lsf.group_hosts(BMGROUP) == [
        "cub22-oversubscribed03",
        "cub22-oversubscribed01",
        "cub22-oversubscribed02",
    ]


def test_free_memory_takes_the_largest_healthy_host():
    assert lsf.free_memory_mb(LSLOAD) == pytest.approx(8192.0)
    assert lsf.free_memory_mb(LSLOAD_BIG) == pytest.approx(1782579.2)


def test_free_slots_skips_hosts_that_are_not_ok():
    assert lsf.free_slots(BHOSTS) == 30


def test_gpu_detection_distinguishes_real_rows_from_the_cold_cache_message():
    assert lsf.has_gpus(BHOSTS_GPU)
    assert lsf.free_gpus(BHOSTS_GPU) == 1
    assert not lsf.has_gpus(BHOSTS_GPU_NONE)


def _fake_cluster(lsload=LSLOAD, gpu=BHOSTS_GPU_NONE):
    def run(command):
        if command[0] == "bqueues":
            return BQUEUES_OVERSUBSCRIBED
        if command[0] == "bmgroup":
            return BMGROUP
        if command[0] == "lsload":
            return lsload
        if command[:2] == ["bhosts", "-gpu"]:
            return gpu
        return BHOSTS

    return run


def test_memory_request_above_free_memory_is_reported_not_silently_pended():
    capacity = lsf.queue_capacity("oversubscribed", run=_fake_cluster())

    assert capacity.known
    problems = lsf.request_problems(capacity, memory_gb=16, cores=4, gpu=None)

    assert len(problems) == 1
    assert "pend indefinitely" in problems[0]
    assert lsf.request_problems(capacity, memory_gb=6, cores=4, gpu=None) == []


def test_cpu_job_on_a_gpu_queue_is_reported_before_esub_rejects_it():
    capacity = lsf.queue_capacity(
        "training-normal", run=_fake_cluster(lsload=LSLOAD_BIG, gpu=BHOSTS_GPU)
    )

    problems = lsf.request_problems(capacity, memory_gb=32, cores=8, gpu=None)

    assert len(problems) == 1
    assert "GPU-backed" in problems[0]
    assert lsf.request_problems(capacity, memory_gb=32, cores=8, gpu="-") == []


def test_unknown_capacity_never_blocks_a_submission():
    capacity = lsf.queue_capacity("mystery", run=lambda command: "")

    assert not capacity.known
    assert lsf.request_problems(capacity, memory_gb=999, cores=99, gpu=None) == []


def test_choose_queue_prefers_a_queue_that_fits_the_request(monkeypatch):
    monkeypatch.setattr(lsf, "CPU_QUEUES", ("oversubscribed", "spare"))

    assert (
        lsf.choose_queue(memory_gb=6, cores=4, gpu=None, run=_fake_cluster())
        == "oversubscribed"
    )


def _two_queue_cluster():
    """A small CPU queue and a large GPU queue, as on the real cluster."""

    def run(command):
        target = command[-1]
        gpu_queue = "big" in (command[2] if command[0] == "bqueues" else target)
        if command[0] == "bqueues":
            return "HOSTS:  big_hosts/\n" if gpu_queue else BQUEUES_OVERSUBSCRIBED
        if command[0] == "bmgroup":
            return (
                "GROUP_NAME    HOSTS\nbig  farm-gpu0301\n"
                if command[1] == "big_hosts"
                else BMGROUP
            )
        big = any("farm-gpu" in part for part in command)
        if command[0] == "lsload":
            return LSLOAD_BIG if big else LSLOAD
        if command[:2] == ["bhosts", "-gpu"]:
            return BHOSTS_GPU if big else BHOSTS_GPU_NONE
        return BHOSTS

    return run


def test_cpu_work_too_large_for_the_cpu_queue_spills_onto_a_gpu_queue(monkeypatch):
    monkeypatch.setattr(lsf, "CPU_QUEUES", ("oversubscribed",))
    monkeypatch.setattr(lsf, "GPU_QUEUES", ("big",))
    run = _two_queue_cluster()

    # 6G fits the cheap CPU queue, so no GPU is reserved.
    assert lsf.choose_queue(memory_gb=6, cores=4, gpu=None, run=run) == "oversubscribed"
    # 64G fits nowhere on CPU, so the GPU-backed queue is the only way to run.
    assert lsf.choose_queue(memory_gb=64, cores=4, gpu=None, run=run) == "big"


def test_choose_queue_falls_back_rather_than_blocking_the_submission(monkeypatch):
    monkeypatch.setattr(lsf, "CPU_QUEUES", ("oversubscribed", "spare"))

    # 64G fits nowhere, and an uninspectable cluster answers nothing at all;
    # both still yield a queue, leaving the verdict to the caller's preflight.
    assert (
        lsf.choose_queue(memory_gb=64, cores=4, gpu=None, run=_fake_cluster())
        == "oversubscribed"
    )
    assert (
        lsf.choose_queue(memory_gb=6, cores=4, gpu=None, run=lambda command: "")
        == "oversubscribed"
    )
