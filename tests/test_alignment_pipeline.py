import copy
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from app import pipeline


class RecordingModel:
    def __init__(self, model_type="wav2vec2"):
        self.config = types.SimpleNamespace(
            model_type=model_type, _name_or_path="fixture/wav2vec2"
        )
        self.inputs = []

    def __call__(self, waveform):
        self.inputs.append(waveform.clone())
        return types.SimpleNamespace(logits=waveform)


class AlignmentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.saved_models = pipeline._align_models.copy()
        self.saved_used = pipeline._align_models_last_used.copy()
        pipeline._align_models.clear()
        pipeline._align_models_last_used.clear()
        self.addCleanup(self.restore_cache)

    def restore_cache(self):
        pipeline._align_models.clear()
        pipeline._align_models.update(self.saved_models)
        pipeline._align_models_last_used.clear()
        pipeline._align_models_last_used.update(self.saved_used)

    def test_quiet_windows_reach_model_with_feature_extractor_normalization(self):
        model = RecordingModel()
        extractor = Wav2Vec2FeatureExtractor(do_normalize=True)
        audio = np.array([.0001, -.0002, .0003, -.0004,
                          .3, -.2, .7, -.5], dtype=np.float32)
        original = audio.copy()
        transcript = {"language": "ru", "segments": [{
            "start": 0., "end": .0005, "text": "unchanged"
        }]}

        def align_windows(segments, selected_model, metadata, source, device, **kwargs):
            for window in (source[:4], source[4:]):
                selected_model(torch.from_numpy(window).unsqueeze(0))
            return {"segments": copy.deepcopy(segments)}

        with patch.object(pipeline.whisperx, "load_align_model", return_value=(
            model, {"type": "huggingface", "language": "ru"}
        )) as load, patch("transformers.AutoFeatureExtractor.from_pretrained",
                         return_value=extractor) as metadata_load, \
                patch.object(pipeline.whisperx, "align", side_effect=align_windows), \
                patch.object(pipeline, "clear_gpu_memory"), \
                patch.object(pipeline, "_ensure_eviction_thread"):
            result = pipeline.align(audio, copy.deepcopy(transcript))
            pipeline.align(audio, copy.deepcopy(transcript))
        for index, window in enumerate((original[:4], original[4:])):
            expected = extractor.zero_mean_unit_var_norm([window], None)[0]
            torch.testing.assert_close(model.inputs[index][0], torch.from_numpy(expected))
        np.testing.assert_array_equal(audio, original)
        self.assertEqual(result["segments"], transcript["segments"])
        self.assertEqual(load.call_count, 1)
        self.assertEqual(metadata_load.call_count, 1)

    def test_metadata_failure_returns_coarse_segments_and_does_not_cache_model(self):
        model = RecordingModel()
        transcript = {"language": "ru", "segments": [{
            "start": 1., "end": 27., "text": "unchanged"
        }]}
        with patch.object(pipeline.whisperx, "load_align_model", return_value=(
            model, {"type": "huggingface", "language": "ru"}
        )), patch("transformers.AutoFeatureExtractor.from_pretrained",
                  side_effect=OSError("not cached")), \
                patch.object(pipeline.whisperx, "align") as align, \
                self.assertLogs(pipeline.logger, level="WARNING") as logs:
            result = pipeline.align(np.zeros(32, dtype=np.float32), transcript)
        self.assertIs(result, transcript)
        self.assertNotIn("ru", pipeline._align_models)
        self.assertIn("not cached", logs.output[0])
        align.assert_not_called()

    def test_qwen_path_does_not_load_wav2vec_metadata(self):
        qwen = types.ModuleType("app.qwen3_backend")
        qwen.align = unittest.mock.Mock(return_value={"segments": []})
        for backend_flag in ({"_asr_backend": "qwen3"}, {"_qwen_align": True}):
            with self.subTest(flag=backend_flag):
                with patch.dict("sys.modules", {"app.qwen3_backend": qwen}), \
                        patch.object(pipeline, "load_align_model") as load, \
                        patch.object(pipeline, "clear_gpu_memory"):
                    result = pipeline.align(np.zeros(32, dtype=np.float32), backend_flag)
                self.assertIs(result, qwen.align.return_value)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
