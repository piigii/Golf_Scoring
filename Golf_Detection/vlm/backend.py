"""Lazy-loaded LocateAnything backends. MLX checkpoints and CUDA weights differ."""
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CUDA_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"


def select_backend(name):
    if name == "auto":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return "metal"
        import torch
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("VLM requires Apple Silicon Metal or NVIDIA CUDA; specify --backend.")
    if name not in {"metal", "cuda"}:
        raise ValueError(f"Unknown VLM backend: {name}")
    return name


def default_model(backend):
    return ROOT / "models" / ("LocateAnything-3B-MLX" if backend == "metal" else "LocateAnything-3B-CUDA")


def model_identity(path):
    """Hash weights and runtime/config files so cached detections cannot mix models."""
    import hashlib
    files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
    hashes = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(file.relative_to(path))] = digest.hexdigest()
    manifest = path / "download_manifest.json"
    info = json.loads(manifest.read_text()) if manifest.exists() else {}
    return {"model": info.get("repo_id", path.name), "revision": info.get("revision"), "sha256": hashes}


def load_backend(name, model, image_token_limit=1024):
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"Missing model: {model}. Run vlm/download_model.py --backend {name}.")
    return MetalBackend(model, image_token_limit) if name == "metal" else CudaBackend(model, image_token_limit)


class MetalBackend:
    name = "metal"

    def __init__(self, path, image_token_limit):
        import mlx.core as mx
        from mlx_vlm import load
        from checkpoint_compat import install_checkpoint_aliases
        if not mx.metal.is_available():
            raise RuntimeError("Metal unavailable; use native ARM Python on Apple Silicon.")
        mx.set_wired_limit(min(8 * 1024**3, mx.device_info()["max_recommended_working_set_size"]))
        mx.set_cache_limit(256 * 1024**2)
        install_checkpoint_aliases()
        self.mx = mx
        self.model, self.processor = load(str(path.resolve()), strict=True)
        self.processor.image_processor.in_token_limit = image_token_limit
        self.hardware = mx.device_info().get("device_name", platform.machine())
        self.dtype = "bfloat16"

    def predict(self, image, question, max_tokens=48, mode="hybrid"):
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs
        prompt = apply_chat_template(self.processor, self.model.config, question, num_images=1)
        inputs = prepare_inputs(self.processor, images=[image], prompts=prompt)
        input_ids = inputs.pop("input_ids")
        inputs.pop("attention_mask", None)
        tokens = self.model.pbd_generate(input_ids, generation_mode=mode, max_tokens=max_tokens, **inputs)
        return self.processor.decode(tokens, skip_special_tokens=False)

    def clear_cache(self):
        self.mx.clear_cache()

    def memory(self):
        return {"mlx_peak_memory_bytes": self.mx.get_peak_memory()}


class CudaBackend:
    name = "cuda"

    def __init__(self, path, image_token_limit):
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; check your NVIDIA driver and PyTorch CUDA installation.")
        self.torch = torch
        self.dtype_value = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.dtype = str(self.dtype_value).removeprefix("torch.")
        self.hardware = torch.cuda.get_device_name(0)
        options = {"trust_remote_code": True, "local_files_only": True}
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), **options)
        self.processor = AutoProcessor.from_pretrained(str(path), **options)
        self.processor.image_processor.in_token_limit = image_token_limit
        config = AutoConfig.from_pretrained(str(path), **options)
        for section in (config, config.text_config, config.vision_config):
            section._attn_implementation = "sdpa"
        self.model = AutoModel.from_pretrained(
            str(path), config=config, torch_dtype=self.dtype_value, attn_implementation="sdpa", **options
        ).to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()

    def predict(self, image, question, max_tokens=48, mode="hybrid"):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            response = self.model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"].to(self.dtype_value),
                attention_mask=inputs["attention_mask"], image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer, max_new_tokens=max_tokens, generation_mode=mode,
                use_cache=True, do_sample=False, verbose=False)
        self.torch.cuda.synchronize()
        answer = response[0] if isinstance(response, tuple) else response
        if not isinstance(answer, str):
            raise TypeError("Unexpected LocateAnything response; expected decoded text.")
        return answer

    def clear_cache(self):
        self.torch.cuda.empty_cache()

    def memory(self):
        return {"cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved()}
