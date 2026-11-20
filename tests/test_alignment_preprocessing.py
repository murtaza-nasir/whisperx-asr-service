import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from app.alignment_preprocessing import (
    NormalizedAlignmentModel, normalize_waveform, prepare_alignment_model,
)


class WaveformNormalizationTests(unittest.TestCase):
    def test_matches_feature_extractor_on_cpu_and_cuda_without_mutating_input(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        print("Normalization reference devices: " + ", ".join(devices), flush=True)
        for device in devices:
            for dtype in (torch.float16, torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    windows = torch.tensor([
                        [.0001, -.0002, .0003, -.0004],
                        [.3, -.2, .7, -.5],
                        [1.2, 1.7, 1.4, 1.9],
                    ], dtype=dtype, device=device)
                    original = windows.clone()
                    expected = Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm(
                        list(windows.float().cpu().numpy()), None
                    )
                    actual = normalize_waveform(windows)
                    torch.testing.assert_close(actual, torch.tensor(
                        np.array(expected), device=device
                    ), atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(windows, original)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertEqual(actual.device, windows.device)

    def test_matches_installed_whisperx_short_window_padding_contract(self):
        # WhisperX pads windows shorter than 400 samples before the HF model
        # call, without providing a valid-length mask. Match that actual input;
        # do not infer padding by trimming legitimate trailing zero samples.
        window = torch.tensor([[.001, -.002, .003]])
        padded = torch.nn.functional.pad(window, (0, 400 - window.shape[-1]))
        expected = Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm(
            [padded[0].numpy()], None
        )[0]
        actual = normalize_waveform(padded)
        self.assertEqual(actual.shape, (1, 400))
        torch.testing.assert_close(actual[0], torch.from_numpy(expected))

    def test_constant_singleton_and_empty_inputs_remain_finite(self):
        for waveform in (torch.zeros(2, 32), torch.full((1, 32), 3.),
                         torch.tensor([[2.]]), torch.empty(2, 0)):
            with self.subTest(shape=waveform.shape):
                actual = normalize_waveform(waveform)
                self.assertEqual(actual.shape, waveform.shape)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_rejects_nonfinite_integer_and_scalar_inputs(self):
        for waveform in (torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                         torch.tensor([1, 2]), torch.tensor(1.)):
            with self.subTest(waveform=waveform):
                with self.assertRaises(ValueError):
                    normalize_waveform(waveform)

    def test_adapter_preserves_model_output_and_call_arguments(self):
        model = Mock(return_value=object())
        adapter = NormalizedAlignmentModel(model)
        waveform = torch.tensor([[.001, -.001]])
        result = adapter(waveform, "extra", return_dict=True)
        self.assertIs(result, model.return_value)
        self.assertEqual(model.call_args.args[1], "extra")
        self.assertEqual(model.call_args.kwargs, {"return_dict": True})
        torch.testing.assert_close(model.call_args.args[0], normalize_waveform(waveform))


class AlignmentModelPreparationTests(unittest.TestCase):
    def model(self, **overrides):
        config = dict(model_type="wav2vec2", _name_or_path="fixture/model")
        config.update(overrides)
        return types.SimpleNamespace(config=types.SimpleNamespace(**config))

    def test_normalization_respects_feature_extractor_flag_and_cache(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                model = self.model()
                with patch("transformers.AutoFeatureExtractor.from_pretrained",
                           return_value=Wav2Vec2FeatureExtractor(do_normalize=enabled)) as load:
                    prepared = prepare_alignment_model(model, {"type": "huggingface"}, "/cache")
                load.assert_called_once_with("fixture/model", cache_dir="/cache", local_files_only=True)
                if enabled:
                    self.assertIsInstance(prepared, NormalizedAlignmentModel)
                    self.assertIs(prepared.model, model)
                else:
                    self.assertIs(prepared, model)

    def test_other_model_backends_do_not_load_metadata(self):
        for metadata, model in (({"type": "torchaudio"}, self.model()),
                                ({"type": "huggingface"}, self.model(model_type="other"))):
            with self.subTest(metadata=metadata, config=model.config):
                with patch("transformers.AutoFeatureExtractor.from_pretrained") as load:
                    self.assertIs(prepare_alignment_model(model, metadata, "/cache"), model)
                load.assert_not_called()

    def test_missing_metadata_is_not_silently_ignored(self):
        with patch("transformers.AutoFeatureExtractor.from_pretrained",
                   side_effect=OSError("not cached")):
            with self.assertRaisesRegex(OSError, "not cached"):
                prepare_alignment_model(self.model(), {"type": "huggingface"}, "/cache")

    def test_rejects_incompatible_metadata(self):
        for extractor in (object(), Wav2Vec2FeatureExtractor(sampling_rate=8000),
                          Wav2Vec2FeatureExtractor(do_normalize="false")):
            with self.subTest(extractor=extractor):
                with patch("transformers.AutoFeatureExtractor.from_pretrained",
                           return_value=extractor):
                    with self.assertRaises(ValueError):
                        prepare_alignment_model(self.model(), {"type": "huggingface"}, "/cache")
        with self.assertRaisesRegex(ValueError, "identity"):
            prepare_alignment_model(self.model(_name_or_path=""), {"type": "huggingface"}, "/cache")


if __name__ == "__main__":
    unittest.main()
