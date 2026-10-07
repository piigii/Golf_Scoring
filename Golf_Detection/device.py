"""Shared PyTorch device selection for Apple Metal, NVIDIA CUDA and CPU."""
import torch


def select_device(name="auto"):
    mps = getattr(torch.backends, "mps", None)
    has_mps = mps is not None and mps.is_available()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device: {name}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a CUDA PyTorch wheel and check the NVIDIA driver.")
    if name == "mps" and not has_mps:
        raise RuntimeError("Apple Metal/MPS unavailable. Use native ARM Python on Apple Silicon, or --device cpu.")
    return torch.device(name)
