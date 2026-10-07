"""Device routing, CUDA adapter contract, and optional real GPU smoke tests."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from device import select_device
from model import EventDetector

VLM = Path(__file__).resolve().parents[1] / "vlm"
sys.path.insert(0, str(VLM))
from backend import CudaBackend, select_backend
from infer import parse_coordinates


class DeviceTests(unittest.TestCase):
    def test_auto_priority_and_unavailable_device_errors(self):
        for cuda, mps, expected in [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]:
            with patch("torch.cuda.is_available", return_value=cuda), patch("torch.backends.mps.is_available", return_value=mps):
                self.assertEqual(select_device().type, expected)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                select_device("cuda")
        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS unavailable"):
                select_device("mps")

    def test_vlm_auto_platform_routing(self):
        with patch("backend.platform.system", return_value="Darwin"), patch("backend.platform.machine", return_value="arm64"):
            self.assertEqual(select_backend("auto"), "metal")
        with patch("backend.platform.system", return_value="Linux"), patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(select_backend("auto"), "cuda")

    def test_cuda_adapter_uses_model_specific_processor_and_preserves_text(self):
        # Test only the official API contract, without allocating a CUDA model.
        engine = CudaBackend.__new__(CudaBackend)
        engine.torch = SimpleNamespace(inference_mode=torch.inference_mode, cuda=Mock())
        engine.dtype_value = torch.float16
        engine.tokenizer = object()
        batch = Mock()
        inputs = {"input_ids": torch.ones((1, 2), dtype=torch.long),
                  "pixel_values": torch.ones((1, 3)), "attention_mask": torch.ones((1, 2)),
                  "image_grid_hws": torch.tensor([[1, 1]])}
        batch.to.return_value = inputs
        engine.processor = Mock(return_value=batch)
        engine.processor.process_vision_info.return_value = (["image"], None)
        engine.model = Mock()
        raw = "<box><1><2><300><400></box>"
        engine.model.generate.return_value = (raw, None, {})
        self.assertEqual(engine.predict("image", "golf club", max_tokens=48), raw)
        call = engine.model.generate.call_args.kwargs
        self.assertEqual(call["pixel_values"].dtype, torch.float16)
        self.assertIs(call["tokenizer"], engine.tokenizer)
        self.assertFalse(call["do_sample"])
        self.assertEqual(call["generation_mode"], "hybrid")
        engine.torch.cuda.synchronize.assert_called_once()
        self.assertEqual(parse_coordinates(raw, 160, 160)[0]["pixels_xyxy"], [.16, .32, 48., 64.])

    def test_coordinate_parser_rejects_invalid_boxes(self):
        self.assertEqual(parse_coordinates("<box><900><2><1><4></box>", 160, 160), [])
        self.assertEqual(parse_coordinates("<box><1><2><1001><4></box>", 160, 160), [])

    def test_swingnet_on_available_gpu(self):
        device = select_device()
        if device.type == "cpu":
            self.skipTest("No accessible CUDA/MPS GPU")
        model = EventDetector(False, 1., 1, 256, dropout=False).to(device).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 2, 3, 32, 32, device=device))
        self.assertEqual(output.device.type, device.type)
        self.assertEqual(tuple(output.shape), (2, 9))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
