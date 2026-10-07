"""Interface tests; synthetic fixtures are never reported as video predictions."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from golf_pipeline import ROOT, event_records, make_events, pose_3d, read_video, render
from model import EventDetector


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cpu_swingnet_without_mobilenet_file(self):
        model = EventDetector(False, 1., 1, 256, dropout=False).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 9))
        self.assertEqual(output.device.type, "cpu")
        self.assertFalse(model.hidden[0].requires_grad)

    def test_events_use_original_frame_and_timestamp(self):
        probabilities = np.zeros((20, 9), dtype=np.float32)
        expected_frames = np.array([0, 2, 4, 6, 8, 10, 12, 19])
        probabilities[expected_frames, np.arange(8)] = .9
        timestamps = np.arange(20) / 30 + .012
        points = np.arange(20 * 17 * 2).reshape(20, 17, 2)
        scores = np.ones((20, 17))
        poses = np.arange(20 * 17 * 3).reshape(20, 17, 3)
        events, ordered = make_events(probabilities, timestamps, points, scores, poses)
        self.assertTrue(ordered)
        for event, frame in zip(events, expected_frames):
            self.assertEqual(event["frame"], frame)
            self.assertEqual(event["time_seconds"], timestamps[frame])
            np.testing.assert_array_equal(event["pose_2d"], points[frame])
            np.testing.assert_array_equal(event["pose_3d_root_relative_m"], poses[frame])

    def test_out_of_order_events_are_reported_without_relabeling(self):
        probabilities = np.zeros((10, 9))
        probabilities[9, 0] = 1
        probabilities[1:8, np.arange(1, 8)] = 1
        events, ordered = event_records(probabilities, np.arange(10) / 30)
        self.assertFalse(ordered)
        self.assertEqual(events[0]["frame"], 9)
        self.assertEqual(events[0]["name"], "Address")

    def test_frame_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same original video frames"):
            make_events(np.zeros((10, 9)), np.arange(10), np.zeros((9, 17, 2)),
                        np.zeros((10, 17)), np.zeros((10, 17, 3)))

    def test_missing_video_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Cannot open"):
            read_video(ROOT / "does_not_exist.mp4")

    def test_portable_golfpose_hrnet_and_decoder_import_without_mmcv_ops(self):
        from mmengine.config import Config
        from golfpose_hrnet import GolfPoseHRNet
        from mmpose.codecs import MSRAHeatmap
        from mmpose.structures.bbox import get_warp_matrix
        config = ROOT / "third_party/GolfPose/configs/mmpose/golfpose_golfer_hrnetw48.py"
        if not config.is_file():
            self.skipTest("GolfPose checkout not present")
        cfg = Config.fromfile(str(config))
        model = GolfPoseHRNet(cfg.model).eval()
        with torch.inference_mode():
            heatmaps = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(heatmaps.shape), (1, 22, 16, 16))
        codec = MSRAHeatmap(input_size=(64, 64), heatmap_size=(16, 16), sigma=3)
        keypoints, scores = codec.decode(heatmaps[0].numpy())
        self.assertEqual(keypoints.shape, (1, 22, 2))
        self.assertEqual(scores.shape, (1, 22))
        warp = get_warp_matrix(np.array([32, 32]), np.array([64, 64]), 0, (64, 64))
        self.assertEqual(warp.shape, (2, 3))

    def test_golfpose_3d_short_and_overlapping_sequences(self):
        # Tiny untrained upstream model, solely to validate adapter interfaces.
        import sys
        repo = ROOT / "third_party/GolfPose"
        if not repo.is_dir():
            self.skipTest("GolfPose checkout not present")
        sys.path.insert(0, str(repo))
        from common.model_cross import MixSTE2
        torch.manual_seed(42)
        model = MixSTE2(num_frame=5, num_joints=17, in_chans=2,
                        embed_dim_ratio=8, depth=1, num_heads=8,
                        mlp_ratio=2., qkv_bias=True, drop_path_rate=0.)
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "synthetic_interface_fixture.bin"
            torch.save({"model_pos": {"module." + k: v for k, v in model.state_dict().items()}}, checkpoint)
            for length in (3, 8):
                points = rng.uniform(0, 160, (length, 17, 2)).astype(np.float32)
                result = pose_3d(points, 160, 160, repo, checkpoint, torch.device("cpu"))
                self.assertEqual(result.shape, (length, 17, 3))
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_allclose(result[:, 0], 0, atol=1e-7)

    def test_renderer_preserves_frame_count_and_exports_eight_events(self):
        import cv2
        frames = [np.zeros((160, 160, 3), dtype=np.uint8) for _ in range(8)]
        points = np.full((8, 17, 2), 80, dtype=np.float32)
        scores = np.ones((8, 17), dtype=np.float32)
        poses = np.zeros((8, 17, 3), dtype=np.float32)
        probabilities = np.zeros((8, 9))
        probabilities[np.arange(8), np.arange(8)] = 1
        events, _ = make_events(probabilities, np.arange(8) / 30, points, scores, poses)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            render(frames, points, scores, poses, events, 30, destination, .3)
            rendered, fps, _, _ = read_video(destination / "annotated.mp4")
            self.assertEqual(len(rendered), 8)
            self.assertEqual(rendered[0].shape, (480, 960, 3))
            self.assertAlmostEqual(fps, 30)
            self.assertEqual(len(list(destination.glob("event_*.jpg"))), 8)
            montage = cv2.imread(str(destination / "events_montage.jpg"))
            self.assertEqual(montage.shape, (2560, 640, 3))


if __name__ == "__main__":
    unittest.main()
