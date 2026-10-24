"""The CUDA load of nano-parakeet casts to bf16 before the model reaches the GPU.

nano_parakeet.from_pretrained (0.2.1) moves the fp32 model to the device and
casts afterwards, so the GPU briefly holds fp32 weights. On a g6f.xlarge's
3 GiB L4 slice that peak is an OOM although the bf16 model fits.
"""

import sys
import types

import pytest
import torch

from speech_to_speech.STT import parakeet_tdt_handler


class _Part:
    def __init__(self, name, log):
        self.name = name
        self.log = log

    def to(self, arg):
        self.log.append((self.name, arg))
        return self


class _FakeModel:
    def __init__(self, log):
        self.log = log
        self.encoder = _Part("encoder", log)
        self.decoder = _Part("decoder", log)
        self.joint = _Part("joint", log)

    def load_state_dict(self, sd, strict):
        return [], []

    def to(self, arg):
        self.log.append(("model", arg))
        return self

    def eval(self):
        return self

    def warmup(self):
        self.log.append(("warmup", None))


@pytest.fixture
def fake_nano(monkeypatch):
    log = []
    pkg = types.ModuleType("nano_parakeet")
    pkg._MODEL_CACHE = {}
    pkg.ParakeetTDT = lambda: _FakeModel(log)

    def from_pretrained(model_name, device, dtype=None):
        log.append(("from_pretrained", device))
        return "library-model"

    pkg.from_pretrained = from_pretrained
    loader = types.ModuleType("nano_parakeet._loader")
    loader.load_nemo_state_dict = lambda path, map_location: {}
    loader.remap_state_dict = lambda sd: sd
    loader.get_bundled_tokenizer_proto = lambda: b""
    monkeypatch.setitem(sys.modules, "nano_parakeet", pkg)
    monkeypatch.setitem(sys.modules, "nano_parakeet._loader", loader)
    monkeypatch.setattr(parakeet_tdt_handler, "_nano_hf_download", lambda model_name: "/fake.nemo")
    monkeypatch.setattr(parakeet_tdt_handler, "_nano_tokenizer", lambda proto: object())
    return pkg, log


def _bf16(monkeypatch, supported):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: supported)


def test_cuda_bf16_load_casts_every_part_before_moving_the_model(fake_nano, monkeypatch):
    _, log = fake_nano
    _bf16(monkeypatch, True)

    parakeet_tdt_handler._load_nano_parakeet("nvidia/parakeet-tdt-0.6b-v3", "cuda")

    move = log.index(("model", "cuda"))
    for part in ("encoder", "decoder", "joint"):
        assert log.index((part, torch.bfloat16)) < move
    assert log[-1] == ("warmup", None)


def test_cuda_bf16_load_shares_the_library_cache_across_pipelines(fake_nano, monkeypatch):
    pkg, log = fake_nano
    _bf16(monkeypatch, True)

    first = parakeet_tdt_handler._load_nano_parakeet("m/x", "cuda")
    second = parakeet_tdt_handler._load_nano_parakeet("m/x", "cuda")

    assert first is second
    assert pkg._MODEL_CACHE[("m/x", "cuda", torch.bfloat16)] is first
    assert log.count(("model", "cuda")) == 1


def test_gpu_without_bf16_uses_the_library_loader(fake_nano, monkeypatch):
    _, log = fake_nano
    _bf16(monkeypatch, False)

    assert parakeet_tdt_handler._load_nano_parakeet("m/x", "cuda") == "library-model"
    assert log == [("from_pretrained", "cuda")]


def test_cpu_uses_the_library_loader(fake_nano, monkeypatch):
    _, log = fake_nano
    _bf16(monkeypatch, True)

    assert parakeet_tdt_handler._load_nano_parakeet("m/x", "cpu") == "library-model"
    assert log == [("from_pretrained", "cpu")]
