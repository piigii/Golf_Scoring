"""Check GPU access and checkpoints in the current environment."""
import importlib.util
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
result = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()}
if importlib.util.find_spec("torch"):
    import torch
    from device import select_device
    result.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                  mps=torch.backends.mps.is_available(), selected_device=str(select_device()))
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
if importlib.util.find_spec("mlx"):
    try:
        import mlx.core as mx
        result["metal"] = mx.metal.is_available()
    except (ImportError, RuntimeError) as error:
        result["metal_error"] = str(error)
result["assets"] = {name: (ROOT / name).exists() for name in [
    "third_party/GolfPose/configs/mmpose/golfpose_golfer_hrnetw48.py",
    "models/swingnet_1800.pth.tar", "models/golfpose_golfer_hrnetw48.pth",
    "models/golfpose_17+5_39.2_32.3_62.8.bin", "vlm/models/LocateAnything-3B-MLX/config.json",
    "vlm/models/LocateAnything-3B-CUDA/config.json"]}
print(json.dumps(result, indent=2))
