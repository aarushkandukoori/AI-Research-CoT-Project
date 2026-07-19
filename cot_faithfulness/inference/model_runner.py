"""GPU probing and model loading with 4-bit quantization."""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class DeviceInfo:
    device_type: str  # cuda, mps, cpu
    device_name: str
    total_vram_gb: float | None
    supports_4bit: bool


def get_device_info() -> DeviceInfo:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo(
            device_type="cuda",
            device_name=props.name,
            total_vram_gb=props.total_memory / (1024**3),
            supports_4bit=True,
        )
    if torch.backends.mps.is_available():
        return DeviceInfo(
            device_type="mps",
            device_name="Apple MPS",
            total_vram_gb=None,
            supports_4bit=False,
        )
    return DeviceInfo(
        device_type="cpu",
        device_name="CPU",
        total_vram_gb=None,
        supports_4bit=False,
    )


def select_load_strategy(model_id: str, device_info: DeviceInfo) -> dict:
    """Choose quantization and dtype based on available hardware."""
    short_name = model_id.split("/")[-1]
    is_large = "7B" in short_name or "7b" in short_name

    if device_info.supports_4bit:
        return {
            "use_4bit": True,
            "dtype": "float16",
            "device_map": "auto",
            "loadable": True,
            "note": "4-bit quantization via bitsandbytes on CUDA",
        }

    if device_info.device_type == "mps":
        # MPS + bnb is unreliable; use fp16 for small models only
        return {
            "use_4bit": False,
            "dtype": "float16",
            "device_map": None,
            "loadable": not is_large,
            "note": "MPS fp16 fallback (4-bit requires CUDA); 7B may OOM",
        }

    # CPU fallback
    return {
        "use_4bit": False,
        "dtype": "float32",
        "device_map": None,
        "loadable": True,
        "note": "CPU fp32 fallback (slow; use Colab T4 + 4-bit for full run)",
    }


def _resolve_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def probe_model_loadable(model_id: str) -> dict:
    """Attempt a lightweight load probe without keeping model in memory."""
    device_info = get_device_info()
    strategy = select_load_strategy(model_id, device_info)
    result = {
        "model_id": model_id,
        "device": device_info.__dict__,
        "strategy": strategy,
        "loadable": strategy["loadable"],
    }
    if not strategy["loadable"]:
        return result

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs: dict = {"dtype": _resolve_dtype(strategy["dtype"])}
        if strategy["use_4bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if not strategy["use_4bit"]:
            target = (
                torch.device("mps")
                if device_info.device_type == "mps"
                else torch.device("cpu")
            )
            model = model.to(target)

        param_device = str(next(model.parameters()).device)
        result["probe_success"] = True
        result["param_device"] = param_device
        result["num_parameters_M"] = sum(p.numel() for p in model.parameters()) / 1e6

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as exc:
        result["probe_success"] = False
        result["error"] = str(exc)
        result["loadable"] = False

    return result


class ModelRunner:
    """Run CoT generations with a loaded HF model."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device_info = get_device_info()
        self.strategy = select_load_strategy(model_id, self.device_info)
        if not self.strategy["loadable"]:
            raise RuntimeError(
                f"Model {model_id} not loadable on {self.device_info.device_type}: "
                f"{self.strategy['note']}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        load_kwargs: dict = {"dtype": _resolve_dtype(self.strategy["dtype"])}
        if self.strategy["use_4bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["low_cpu_mem_usage"] = True

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if not self.strategy["use_4bit"]:
            target = (
                torch.device("mps")
                if self.device_info.device_type == "mps"
                else torch.device("cpu")
            )
            self.model = self.model.to(target)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
