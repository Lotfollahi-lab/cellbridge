import pytest

from cellbridge import lsf


def test_current_cluster_uses_lsid():
    output = "My cluster name is cub22\nMy master name is cub22-srv1\n"

    assert lsf.current_cluster(run=lambda command: output) == "cub22"


def test_unknown_cluster_identity_fails_clearly():
    with pytest.raises(RuntimeError, match="run 'lsid'"):
        lsf.current_cluster(run=lambda command: "")


@pytest.mark.parametrize(
    "cluster,gmem,expected",
    [
        ("cub22", None, "oversubscribed"),
        ("cub22", 40_000, "training-normal"),
        ("tiger22", None, "oversubscribed"),
        ("tiger22", 40_000, "training-normal"),
    ],
)
def test_auto_queue_follows_resource_type(cluster, gmem, expected):
    assert lsf.resolve_queue(cluster, "auto", gmem=gmem) == expected


def test_explicit_queue_must_match_resource_type():
    with pytest.raises(ValueError, match="requires --gmem"):
        lsf.resolve_queue("cub22", "training-normal", gmem=None)
    with pytest.raises(ValueError, match="cannot be used with --gmem"):
        lsf.resolve_queue("cub22", "oversubscribed", gmem=1000)


def test_non_batch_and_unknown_queues_fail_with_available_choices():
    with pytest.raises(ValueError, match="not a batch-compute queue"):
        lsf.resolve_queue("cub22", "gpu-interactive", gmem=1000)
    with pytest.raises(ValueError, match="batch queues: oversubscribed"):
        lsf.resolve_queue("cub22", "training-parallel", gmem=1000)


def test_tiger_only_queue_from_cub_names_the_required_head_node():
    with pytest.raises(RuntimeError, match="ssh tiger22-head2"):
        lsf.resolve_cluster("auto", "training-parallel", "cub22")


def test_shared_queue_stays_on_the_current_cluster():
    assert lsf.resolve_cluster("auto", "training-normal", "cub22") == "cub22"
    assert lsf.resolve_cluster("auto", "training-normal", "tiger22") == "tiger22"


def test_explicit_cross_cluster_target_is_rejected_locally():
    with pytest.raises(RuntimeError, match="this node uses cub22"):
        lsf.resolve_cluster("tiger22", "training-normal", "cub22")


def test_group_is_derived_from_gpu_request():
    assert lsf.group_for(None) == "team361"
    assert lsf.group_for(1000) == "s10396"
