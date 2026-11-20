"""Model-selection/import contracts, runnable on the lightweight CI stubs.

Numeric normalization tests use real PyTorch and Transformers in
tests/test_alignment_preprocessing.py.
"""

import builtins
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.alignment_preprocessing import prepare_alignment_model


@pytest.mark.parametrize("backend,model_type", [
    ("torchaudio", "wav2vec2"), ("huggingface", "other"), ("stub", "wav2vec2"),
])
def test_other_backends_do_not_import_feature_extractor(backend, model_type):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "transformers":
            raise AssertionError("Feature extractor is not needed for this backend")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=guarded_import):
        assert prepare_alignment_model(model, {"type": backend}, "/cache") is model


def test_wav2vec_model_without_identity_fails_before_feature_extractor_import():
    model = SimpleNamespace(config=SimpleNamespace(model_type="wav2vec2"))
    with pytest.raises(ValueError, match="identity"):
        prepare_alignment_model(model, {"type": "huggingface"}, "/cache")
