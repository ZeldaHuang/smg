"""Unit tests for the multimodal identity cache salt (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_vllm_mm_salt.py
"""

import importlib.util
from pathlib import Path

from smg_grpc_proto import vllm_engine_pb2

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "mm_salt.py"
_spec = importlib.util.spec_from_file_location("mm_salt", _MODULE_PATH)
mm_salt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm_salt)


def test_gate_accepts_pixel_payloads():
    mm = vllm_engine_pb2.MultimodalInputs(pixel_values=vllm_engine_pb2.TensorData(dtype="float32"))
    assert mm_salt.has_preprocessed_mm_payload(mm)


def test_gate_accepts_grid_only_payloads():
    # The PD decode leg's form: grid tensors without pixels.
    mm = vllm_engine_pb2.MultimodalInputs(mm_hashes=["h1"])
    mm.model_specific_tensors["image_grid_thw"].CopyFrom(
        vllm_engine_pb2.TensorData(dtype="int64", shape=[1, 3])
    )
    assert mm_salt.has_preprocessed_mm_payload(mm)


def test_gate_rejects_identity_only_payloads():
    # Hashes alone cannot rebuild mm features; this leg takes the salt path.
    mm = vllm_engine_pb2.MultimodalInputs(mm_hashes=["h1"])
    assert not mm_salt.has_preprocessed_mm_payload(mm)


def test_empty_hashes_produce_no_salt():
    assert mm_salt.mm_identity_cache_salt([]) is None


def test_salt_is_deterministic_per_content():
    salt = mm_salt.mm_identity_cache_salt(["h1", "h2"])
    assert salt == "mm:h1,h2"
    assert salt == mm_salt.mm_identity_cache_salt(["h1", "h2"])


def test_different_images_get_different_salts():
    assert mm_salt.mm_identity_cache_salt(["dog"]) != mm_salt.mm_identity_cache_salt(["passport"])


def test_salt_is_order_sensitive():
    # Same images in a different order occupy different placeholder positions.
    assert mm_salt.mm_identity_cache_salt(["h1", "h2"]) != mm_salt.mm_identity_cache_salt(
        ["h2", "h1"]
    )


class _MmConfig:
    """Minimal stand-in for vLLM's MultiModalConfig."""

    def __init__(self, *, language_model_only=False, enable_mm_embeds=False, limit_per_prompt=None):
        self.language_model_only = language_model_only
        self.enable_mm_embeds = enable_mm_embeds
        self.limit_per_prompt = limit_per_prompt or {}

    def get_limit_per_prompt(self, modality):
        # Mirrors MultiModalConfig: the flag zeroes every modality.
        if self.language_model_only:
            return 0
        return self.limit_per_prompt.get(modality, 999)


class _ModelConfig:
    """Minimal stand-in for vLLM's ModelConfig."""

    def __init__(self, is_multimodal_model, supports_multimodal_inputs=None, mm_config=None):
        self.is_multimodal_model = is_multimodal_model
        self.architectures = ["FooForConditionalGeneration"]
        self.multimodal_config = _MmConfig() if is_multimodal_model else None
        if mm_config is not None:
            self.multimodal_config = mm_config
        if supports_multimodal_inputs is not None:
            self.supports_multimodal_inputs = supports_multimodal_inputs


def _fake_vllm_registry(monkeypatch, probe, supported=None, info_method="get_processing_info"):
    """Install a fake ``vllm.multimodal`` whose registry probe is ``probe``
    and whose processing info (reached through ``info_method``: the public
    accessor on vLLM main, the private one on 0.19-0.20) lists ``supported``
    modalities (unknown when None)."""
    import sys
    import types

    vllm_mod = types.ModuleType("vllm")
    mm_mod = types.ModuleType("vllm.multimodal")
    registry = types.SimpleNamespace(supports_multimodal_inputs=probe)
    if supported is not None:
        info = types.SimpleNamespace(get_supported_mm_limits=lambda: dict.fromkeys(supported))
        setattr(registry, info_method, lambda mc, tokenizer=None: info)
    mm_mod.MULTIMODAL_REGISTRY = registry
    vllm_mod.multimodal = mm_mod
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.multimodal", mm_mod)


# --- the property path (vLLM main) ---


def test_engine_accepts_mm_inputs_full_vision_worker():
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True, True))


def test_engine_accepts_mm_inputs_language_model_only():
    # --language-model-only keeps the multimodal architecture but zeroes
    # every modality limit: the engine accepts no multimodal inputs.
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(True, False))


def test_engine_accepts_mm_inputs_text_model():
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(False, False))


def test_engine_accepts_mm_inputs_mm_embeds_only(monkeypatch):
    # enable_mm_embeds under --language-model-only ingests pre-computed
    # embeddings but has no encoder for pixel payloads; vLLM counts it as
    # accepting mm inputs.
    flagged = _ModelConfig(
        True,
        True,
        mm_config=_MmConfig(language_model_only=True, enable_mm_embeds=True),
    )
    assert not mm_salt.engine_accepts_mm_inputs(flagged)
    # With embeds on, "no pixels" holds only when every pixel modality the
    # model supports is at 0: an image-only model with image=0 has no use
    # for its tower, a model that also takes video keeps it.
    _fake_vllm_registry(monkeypatch, lambda mc: True, supported=["image"])
    image_only = _ModelConfig(
        True, True, mm_config=_MmConfig(enable_mm_embeds=True, limit_per_prompt={"image": 0})
    )
    assert not mm_salt.engine_accepts_mm_inputs(image_only)
    _fake_vllm_registry(monkeypatch, lambda mc: True, supported=["image", "video"])
    for limits in ({"image": 0}, {"image": 0, "audio": 0}):
        limited = _ModelConfig(
            True, True, mm_config=_MmConfig(enable_mm_embeds=True, limit_per_prompt=limits)
        )
        assert mm_salt.engine_accepts_mm_inputs(limited), limits
    # vLLM 0.19-0.20 reach the same info through the private constructor.
    _fake_vllm_registry(
        monkeypatch, lambda mc: True, supported=["image"], info_method="_create_processing_info"
    )
    assert not mm_salt.engine_accepts_mm_inputs(image_only)
    # The supported set unknown (no registry answer): the tower is assumed.
    _fake_vllm_registry(monkeypatch, lambda mc: True)
    assert mm_salt.engine_accepts_mm_inputs(image_only)


# --- the registry fallback (vLLM 0.19-0.20, no ModelConfig property) ---


def test_engine_accepts_mm_inputs_registry_fallback(monkeypatch):
    _fake_vllm_registry(monkeypatch, lambda mc: True)
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))
    _fake_vllm_registry(monkeypatch, lambda mc: False)
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))


def test_engine_accepts_mm_inputs_registry_fallback_language_model_only(monkeypatch):
    # On vLLM 0.19-0.20 the registry's check reads the zeroed limits and
    # answers False; is_multimodal_model alone would have said True.
    _fake_vllm_registry(monkeypatch, lambda mc: False)
    lmo = _ModelConfig(True, mm_config=_MmConfig(language_model_only=True))
    assert not mm_salt.engine_accepts_mm_inputs(lmo)


def test_engine_accepts_mm_inputs_text_model_skips_the_probe(monkeypatch):
    def probe(mc):
        raise AssertionError("text models must not reach the registry probe")

    _fake_vllm_registry(monkeypatch, probe)
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(False))


def test_engine_accepts_mm_inputs_registry_without_processor(monkeypatch, caplog):
    # A registry without a processor entry for the architecture (a vLLM
    # nightly, say) raises ValueError. Unknown means multimodal, the safe
    # side: the engine rejects what it truly cannot take, while a "no
    # vision" answer would take a healthy decode worker out of service for
    # every PD image request. One warning names the architecture.
    def probe(mc):
        raise ValueError("no processor")

    _fake_vllm_registry(monkeypatch, probe)
    mm_salt._warned_architectures.clear()
    with caplog.at_level("WARNING"):
        assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))
        assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))
    warnings = [r for r in caplog.records if "FooForConditionalGeneration" in r.getMessage()]
    assert len(warnings) == 1
    assert "no processor" in warnings[0].getMessage()
    # --language-model-only is read off the config itself, before any probe:
    # an unknown registry answer never puts such a worker on the vision side.
    lmo = _ModelConfig(True, mm_config=_MmConfig(language_model_only=True))
    assert not mm_salt.engine_accepts_mm_inputs(lmo)


def test_engine_accepts_mm_inputs_registry_failure_keeps_architecture_answer(monkeypatch):
    # An unexpected probe failure must not blind a healthy full-vision worker.
    def probe(mc):
        raise RuntimeError("registry broken")

    _fake_vllm_registry(monkeypatch, probe)
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))


def test_engine_accepts_mm_inputs_without_vllm(monkeypatch):
    # Engine-free context: the architecture is all we have.
    import sys

    monkeypatch.setitem(sys.modules, "vllm", None)
    assert mm_salt.engine_accepts_mm_inputs(_ModelConfig(True))
    assert not mm_salt.engine_accepts_mm_inputs(_ModelConfig(False))
