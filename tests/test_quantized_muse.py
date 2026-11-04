"""Loading-contract tests; fake loader, real CPU torch. No GPU claim."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from odvla.vla.muse_context import FrozenMuseTokens


class QuantizedMuseTests(unittest.TestCase):
    def load(self, quantization, *, advertise_4bit=True):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.text = torch.nn.Module()
                self.text.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])
                self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2))
                self.is_loaded_in_4bit = advertise_4bit

        calls = {}
        def from_pretrained(path, **kwargs):
            calls.update(kwargs)
            return Backbone()
        fake = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
            AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=from_pretrained),
            BitsAndBytesConfig=lambda **kwargs: SimpleNamespace(**kwargs))
        with TemporaryDirectory() as temp, patch.dict(sys.modules, {"transformers": fake}):
            Path(temp, "processor_config.json").write_text(json.dumps({"fixture": True}))
            model = FrozenMuseTokens(temp, 1, quantization=quantization)
        return model, calls

    def test_existing_bf16_loading_stays_unquantized(self):
        model, calls = self.load(None)
        try:
            self.assertNotIn("quantization_config", calls)
            self.assertEqual(calls["device_map"], "auto")
            self.assertFalse(any(p.requires_grad for p in model.model.parameters()))
        finally:
            model.close()

    def test_nf4_is_explicit_bf16_compute_and_single_cuda_device(self):
        model, calls = self.load("nf4")
        try:
            config = calls["quantization_config"]
            self.assertTrue(config.load_in_4bit)
            self.assertEqual(config.bnb_4bit_quant_type, "nf4")
            self.assertEqual(config.bnb_4bit_compute_dtype, torch.bfloat16)
            self.assertEqual(config.bnb_4bit_quant_storage, torch.bfloat16)
            self.assertIn("vision_model", config.llm_int8_skip_modules)
            self.assertIn(r".*\.vision_tower", config.llm_int8_skip_modules)
            self.assertIn(r".*\.vision_adapter", config.llm_int8_skip_modules)
            self.assertIn(r".*\.vision_projection", config.llm_int8_skip_modules)
            self.assertEqual(calls["device_map"], {"": 0})
            self.assertFalse(any(p.requires_grad for p in model.model.parameters()))
        finally:
            model.close()

    def test_nf4_cannot_silently_fall_back_to_unquantized(self):
        with self.assertRaisesRegex(ValueError, "not loaded in 4-bit"):
            self.load("nf4", advertise_4bit=False)


if __name__ == "__main__":
    unittest.main()
