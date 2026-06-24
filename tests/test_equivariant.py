"""Tests for the equivariant model.

build_knn_graph is a pure coords->graph helper and is tested unconditionally.
The model itself requires the optional `flash-eq` dependency, so those tests
skip gracefully when it is not installed (``pip install 'sgnm[equivariant]'``).
"""
import importlib.util

import pytest
import torch

from sgnm.equivariant import EquivariantReactivityModel, build_knn_graph

HAS_FLASH_EQ = importlib.util.find_spec("flash_eq") is not None


# --- build_knn_graph: pure, no flash-eq required --------------------------


def test_knn_nearest_neighbor_on_a_line():
    coords = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    src, dst = build_knn_graph(coords, k=1)
    # Each (dst, src) edge points a node to its single nearest neighbor.
    assert list(zip(dst.tolist(), src.tolist())) == [(0, 1), (1, 0), (2, 1), (3, 2)]


def test_knn_edge_count():
    coords = torch.randn(10, 3)
    src, dst = build_knn_graph(coords, k=3)
    assert src.numel() == dst.numel() == 10 * 3


def test_knn_k_clamped_to_n_minus_one():
    coords = torch.randn(4, 3)
    src, _ = build_knn_graph(coords, k=99)
    assert src.numel() == 4 * 3  # k clamped to n-1


def test_knn_no_self_loops():
    coords = torch.randn(8, 3)
    src, dst = build_knn_graph(coords, k=4)
    assert all(s != d for s, d in zip(src.tolist(), dst.tolist()))


# --- model: requires flash-eq --------------------------------------------


@pytest.mark.skipif(not HAS_FLASH_EQ, reason="flash-eq not installed")
def test_construct_sets_out_channels():
    model = EquivariantReactivityModel(out_channels=3)
    assert model.out_proj.out_features == 3


@pytest.mark.skipif(not HAS_FLASH_EQ, reason="flash-eq not installed")
def test_checkpoint_roundtrip(tmp_path):
    import sgnm

    model = EquivariantReactivityModel(out_channels=2)
    path = tmp_path / "eq.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "init_kwargs": model._init_kwargs,
            "model_type": "EquivariantReactivityModel",
        },
        path,
    )
    loaded = sgnm.load(str(path))
    assert isinstance(loaded, EquivariantReactivityModel)
    for (ka, va), (kb, vb) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
