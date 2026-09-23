"""Engine-free multimodal helpers for tensor-stripped (PD decode) legs."""

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Architectures whose registry probe failed, each warned about once.
_warned_architectures: set[str] = set()


def engine_accepts_mm_inputs(model_config) -> bool:
    """Whether the engine runs a vision encoder for pixel payloads.

    ``is_multimodal_model`` reports the architecture only: it stays true
    under ``--language-model-only``, which drops the encoder, and for an
    ``enable_mm_embeds``-only engine, which ingests embeddings but encodes
    no pixels. The router reads the answer as the worker's ``supports_vision``
    label, so it comes from vLLM's own check: the ``supports_multimodal_inputs``
    property where ``ModelConfig`` has it, the registry's method otherwise.
    """
    mm_config = getattr(model_config, "multimodal_config", None)
    # --language-model-only is a fact of the config, read before any probe
    # whose answer might be unknown.
    if mm_config is not None and getattr(mm_config, "language_model_only", False):
        return False
    supports = getattr(model_config, "supports_multimodal_inputs", None)
    if supports is None:
        supports = _registry_supports_multimodal_inputs(model_config)
    if not supports:
        return False
    # vLLM counts enable_mm_embeds-only engines as accepting multimodal
    # inputs (they ingest embeddings), but they have no encoder for the pixel
    # payloads supports_vision speaks for.
    return not _mm_embeds_only(model_config)


def _registry_supports_multimodal_inputs(model_config) -> bool:
    """vLLM 0.19-0.20 fallback: the check lives on the multimodal registry."""
    if not model_config.is_multimodal_model:
        return False
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY
    except ImportError:
        # Engine-free context (unit tests): the architecture is all we have.
        return True
    try:
        return bool(MULTIMODAL_REGISTRY.supports_multimodal_inputs(model_config))
    except Exception as e:  # noqa: BLE001 - unknown means multimodal, the safe side
        # A registry without an entry for the architecture (ValueError) or
        # any other probe failure: the engine rejects what it truly cannot
        # take, while a "no vision" answer would take a healthy full-vision
        # worker out of service for every PD image request.
        architecture = _architecture_name(model_config)
        if architecture not in _warned_architectures:
            _warned_architectures.add(architecture)
            logger.warning(
                "supports_multimodal_inputs probe failed for %s (%s); reporting the "
                "architecture's multimodal capability",
                architecture,
                e,
            )
        return True


def _architecture_name(model_config) -> str:
    architectures = getattr(model_config, "architectures", None) or []
    return ",".join(architectures) or str(getattr(model_config, "architecture", "unknown"))


def _mm_embeds_only(model_config) -> bool:
    """Whether the engine ingests only pre-computed embeddings (no pixels).

    True when ``enable_mm_embeds`` is on and every modality the model
    supports has a limit of 0: an image-only model with ``image=0`` has no
    use for its tower. The supported set comes from the registry; a limit
    on one modality of a model that supports others is a limit, not the
    absence of the tower, and an unknown supported set assumes the tower.
    """
    mm_config = getattr(model_config, "multimodal_config", None)
    if mm_config is None or not getattr(mm_config, "enable_mm_embeds", False):
        return False
    get_limit = getattr(mm_config, "get_limit_per_prompt", None)
    supported = _supported_modalities(model_config)
    if get_limit is None or not supported:
        return False
    return all(get_limit(modality) == 0 for modality in supported)


def _supported_modalities(model_config) -> list[str]:
    """The modalities the model's processor takes, per vLLM's registry;
    empty when there is no registry to ask."""
    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY

        return list(MULTIMODAL_REGISTRY.get_supported_mm_limits(model_config))
    except Exception:  # noqa: BLE001 - unknown means the tower is assumed
        return []


def has_preprocessed_mm_payload(mm_inputs) -> bool:
    """True when the payload carries tensors the preprocessed path can use.

    A grid-only payload (model-specific tensors, no pixels) is the PD decode
    leg's form; a bare identity payload (hashes only) is not preprocessable
    and falls back to the cache-salt path.
    """
    return mm_inputs.HasField("pixel_values") or bool(mm_inputs.model_specific_tensors)


def mm_identity_cache_salt(mm_hashes: Sequence[str]) -> str | None:
    """Fold per-image content hashes into a deterministic cache salt.

    The PD router strips multimodal tensors from the decode leg (the KV
    arrives via the P/D transfer), keeping only the per-image content hashes.
    Without tensors no ``mm_features`` can be built, so the engine's
    prefix-cache block hashes would carry no image identity — the identity
    rides ``cache_salt`` instead. Deterministic per image content: same-image
    reuse still hits the decode prefix cache, while different images behind
    the same text prefix no longer alias onto each other's KV.
    """
    if not mm_hashes:
        return None
    return "mm:" + ",".join(mm_hashes)
