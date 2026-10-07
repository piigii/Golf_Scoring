"""Portable inference for GolfPose's HRNet configuration/checkpoint.

Loads the installed, unmodified MMPose HRNet backbone in a private namespace.
This avoids importing unrelated detection heads that require compiled MMCV ops.
The head, affine crop, MSRA decoder and flip test follow GolfPose's config.
"""
import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


def hrnet_class():
    import mmpose
    alias = "_golfpose_mmpose_backbones"
    if alias not in sys.modules:
        package = types.ModuleType(alias)
        package.__path__ = [str(Path(mmpose.__file__).parent / "models/backbones")]
        package.__package__ = alias
        sys.modules[alias] = package
    return importlib.import_module(alias + ".hrnet").HRNet


class GolfPoseHRNet(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        if model_config.type != "TopdownPoseEstimator" or model_config.backbone.type != "HRNet":
            raise ValueError("Portable backend supports only GolfPose TopdownPoseEstimator/HRNet.")
        head = model_config.head
        if head.type != "HeatmapHead" or head.get("deconv_out_channels") or head.get("conv_out_channels"):
            raise ValueError("Expected GolfPose HRNet's single-convolution heatmap head.")
        backbone = dict(model_config.backbone)
        backbone.pop("type")
        backbone["init_cfg"] = None
        self.backbone = hrnet_class()(**backbone)
        self.head = torch.nn.Module()
        self.head.final_layer = torch.nn.Conv2d(head.in_channels, head.out_channels, 1)

    def forward(self, images):
        return self.head.final_layer(self.backbone(images)[0])


def infer_hrnet(frames, cfg, info, checkpoint, device):
    from mmpose.codecs import MSRAHeatmap
    from mmpose.structures.bbox import get_warp_matrix

    model = GolfPoseHRNet(cfg.model)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in saved["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    codec_config = dict(cfg.codec)
    codec_config.pop("type")
    codec = MSRAHeatmap(**codec_config)
    input_size = np.asarray(cfg.codec.input_size, dtype=np.float32)
    mean = np.asarray(cfg.model.data_preprocessor.mean, dtype=np.float32)
    std = np.asarray(cfg.model.data_preprocessor.std, dtype=np.float32)
    names = [info["keypoint_info"][i]["name"] for i in range(cfg.model.head.out_channels)]
    flip_indices = [names.index(info["keypoint_info"][i]["swap"] or name)
                    for i, name in enumerate(names)]
    points, scores = [], []
    with torch.inference_mode():
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            center = np.asarray([w / 2, h / 2], dtype=np.float32)
            aspect = input_size[0] / input_size[1]
            # GetBBoxCenterScale uses padding=1.25; TopdownAffine fixes aspect.
            scale = np.asarray([max(w, h * aspect), max(h, w / aspect)], dtype=np.float32) * 1.25
            warp = get_warp_matrix(center, scale, 0, input_size)
            image = cv2.warpAffine(frame, warp, tuple(input_size.astype(int)), flags=cv2.INTER_LINEAR)
            image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) - mean) / std
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
            if cfg.model.test_cfg.get("flip_test", False):
                heatmaps = model(torch.cat([tensor, tensor.flip(-1)], dim=0))
                flipped = heatmaps[1:2, flip_indices].flip(-1)
                if cfg.model.test_cfg.get("shift_heatmap", False):
                    flipped[..., 1:] = flipped[..., :-1].clone()
                heatmap = ((heatmaps[:1] + flipped) / 2)[0].cpu().numpy()
            else:
                heatmap = model(tensor)[0].cpu().numpy()
            keypoints, confidence = codec.decode(heatmap)
            original = keypoints[0] / input_size * scale + center - .5 * scale
            points.append(original)
            scores.append(confidence[0])
            if i % 25 == 0 or i == len(frames) - 1:
                print(f"GolfPose 2D: {i + 1}/{len(frames)}", flush=True)
    return np.asarray(points, dtype=np.float32), np.asarray(scores, dtype=np.float32), names
