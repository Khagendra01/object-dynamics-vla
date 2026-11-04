"""Pinned Muse token interface. No trainable projector is part of the cache."""
from pathlib import Path
import hashlib
import inspect
import numpy as np


def backend_snapshot(torch, model=None):
    """Record numerical settings at the point of use, without modifying them."""
    cuda = torch.backends.cuda
    result = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
              "matmul_allow_tf32": bool(cuda.matmul.allow_tf32),
              "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
              "cudnn_enabled": bool(torch.backends.cudnn.enabled),
              "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
              "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
              "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "cuda_autocast": bool(torch.is_autocast_enabled("cuda")),
              "flash_sdp": bool(cuda.flash_sdp_enabled()),
              "memory_efficient_sdp": bool(cuda.mem_efficient_sdp_enabled()),
              "math_sdp": bool(cuda.math_sdp_enabled())}
    if model is not None:
        result["attention_implementation"] = getattr(model.config, "_attn_implementation", None)
    return result


class FrozenMuseTokens:
    def __init__(self, checkpoint, layer_index=26, *, quantization=None):
        import torch
        import transformers
        self.torch = torch
        self.path = Path(checkpoint)
        self.processor = transformers.AutoProcessor.from_pretrained(str(self.path), trust_remote_code=False)
        factory = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
        if quantization not in (None, "nf4"):
            raise ValueError("quantization must be None or 'nf4'")
        options = {}
        if quantization == "nf4":
            # Keep vision modules unquantized where these architecture names match.
            # The preflight records the actual converted modules; no GGUF is used.
            options["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_storage=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["vision_encoder", "vision_model", "vision_tower",
                    "vision_adapter", "multi_modal_projector", "vision_projection",
                    "vision_projector", "connector", "lm_head",
                    r".*\.vision_tower", r".*\.vision_adapter", r".*\.vision_projection"])
        self.model = factory.from_pretrained(str(self.path), dtype=torch.bfloat16,
            device_map={"": 0} if quantization else "auto", trust_remote_code=False, **options)
        self.quantization = quantization
        if quantization and not getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Requested NF4 model was not loaded in 4-bit")
        self.model.eval().requires_grad_(False)
        count = self.model.config.text_config.num_hidden_layers
        matches = [(n, m) for n, m in self.model.named_modules()
                   if isinstance(m, torch.nn.ModuleList) and len(m) == count and "vision" not in n]
        if len(matches) != 1 or not 0 <= layer_index < count:
            raise ValueError("Cannot uniquely identify requested text decoder block")
        self.module_name, layers = matches[0]
        self.layer_index = layer_index
        self.captured = None
        self.hook = layers[layer_index].register_forward_hook(self._capture)
        self.processor_hash = hashlib.sha256((self.path / "processor_config.json").read_bytes()).hexdigest()

    def _capture(self, module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        if not self.torch.is_tensor(value) or value.ndim != 3:
            raise ValueError("Unexpected Muse layer output")
        self.captured = value.detach().clone()

    def encode(self, image, instruction, *, current_date):
        """One RGB image and instruction; returns all token features and metadata."""
        torch = self.torch
        messages = [{"role": "user", "content": [{"type": "image", "image": image.convert("RGB")},
                                                     {"type": "text", "text": instruction}]}]
        if not isinstance(current_date, str) or not current_date:
            raise ValueError("A fixed chat-template current_date is required for reproducible features")
        batch = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", current_date=current_date)
        batch = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in batch.items()}
        kwargs = {"use_cache": False, "return_dict": True}
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            kwargs["logits_to_keep"] = 1
        self.captured = None
        with torch.inference_mode():
            output = self.model(**batch, **kwargs)
        value = self.captured
        self.captured = None
        if value is None or value.shape[:2] != batch["attention_mask"].shape:
            raise ValueError("Muse output/token mask mismatch")
        if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError("Expected finite BF16 hidden features")
        # NumPy has no portable BF16 dtype: preserve raw bits without conversion.
        bits = value[0].contiguous().view(torch.uint16).cpu().numpy()
        mask = batch["attention_mask"][0].bool().cpu().numpy()
        token_ids = batch["input_ids"][0].cpu().numpy()
        del output
        return {"features_bfloat16_bits": bits, "attention_mask": mask,
                "input_ids": token_ids, "positions": np.arange(len(mask), dtype=np.int32)}

    def close(self):
        self.hook.remove()
