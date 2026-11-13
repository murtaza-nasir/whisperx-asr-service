"""Honor HuggingFace Wav2Vec2 preprocessing at the alignment model boundary.

WhisperX slices the audio into segment windows but calls the CTC model directly,
bypassing its feature extractor. Normalizing here preserves those windows and
leaves the audio used by transcription and diarization untouched.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Standardize each window with the Wav2Vec2 feature extractor formula."""
    if waveform.ndim < 1 or not waveform.is_floating_point():
        raise ValueError("Alignment waveform must be a floating-point time series")
    if not torch.isfinite(waveform).all():
        raise ValueError("Alignment waveform contains non-finite samples")
    values = waveform.to(dtype=torch.float32)
    if values.shape[-1] == 0:
        return values.clone()
    mean = values.mean(dim=-1, keepdim=True)
    variance = values.var(dim=-1, correction=0, keepdim=True)
    return (values - mean) / torch.sqrt(variance + 1e-7)


class NormalizedAlignmentModel:
    """Normalize each WhisperX window immediately before the model call."""

    def __init__(self, model):
        self.model = model

    def __call__(self, waveform, *args, **kwargs):
        return self.model(normalize_waveform(waveform), *args, **kwargs)


def prepare_alignment_model(model, metadata: dict, cache_dir: str):
    """Read the model's cached contract once; other backends pass through.

    WhisperX has already loaded the processor from this cache. Missing or invalid
    metadata is an alignment error, handled by the pipeline's existing fallback,
    rather than permission to silently bypass required normalization.
    """
    if metadata.get("type") != "huggingface":
        return model
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "wav2vec2":
        return model
    model_name = getattr(config, "_name_or_path", None)
    if not model_name:
        raise ValueError("Wav2Vec2 alignment model has no feature extractor identity")
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

    extractor = AutoFeatureExtractor.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=True
    )
    if not isinstance(extractor, Wav2Vec2FeatureExtractor):
        raise ValueError("Alignment model requires a Wav2Vec2 feature extractor")
    if extractor.sampling_rate != 16000:
        raise ValueError("Alignment feature extractor must accept 16000 Hz audio")
    if not isinstance(extractor.do_normalize, bool):
        raise ValueError("Alignment do_normalize must be a boolean")
    if not extractor.do_normalize:
        return model
    logger.info("Enabled per-window waveform normalization for alignment model %s", model_name)
    return NormalizedAlignmentModel(model)
