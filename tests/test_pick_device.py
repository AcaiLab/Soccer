"""Choosing between the GPU and the CPU on a GPU that other people are using."""

import pytest

torch = pytest.importorskip("torch")

from soccer.encoding import pick_device  # noqa: E402


def with_gpu(monkeypatch, free_gb=None, error=None):
    """Pretend a GPU exists with `free_gb` free, or whose memory query raises `error`."""
    def mem_get_info():
        if error:
            raise error
        return free_gb * 1e9, 23e9

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)


def test_an_explicit_device_is_used_as_given():
    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda") == "cuda"


def test_without_a_gpu_the_cpu_is_used(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device("auto") == "cpu"
    assert pick_device(None) == "cpu"


def test_a_gpu_with_room_is_used(monkeypatch):
    with_gpu(monkeypatch, free_gb=8)
    assert pick_device("auto") == "cuda"


def test_a_nearly_full_gpu_is_left_alone(monkeypatch, capsys):
    with_gpu(monkeypatch, free_gb=0.6)
    assert pick_device("auto") == "cpu"
    assert "0.6 GB free" in capsys.readouterr().out


def test_a_gpu_too_full_to_be_queried_falls_back_to_the_cpu(monkeypatch, capsys):
    with_gpu(monkeypatch, error=RuntimeError("CUDA error: out of memory\nCUDA kernel errors may be late"))
    assert pick_device("auto") == "cpu"
    assert "out of memory" in capsys.readouterr().out
