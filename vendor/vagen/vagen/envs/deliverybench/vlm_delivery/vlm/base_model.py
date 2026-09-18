# vlm/base_model.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Union, List, Any
import base64, io, os, time
import numpy as np
from PIL import Image
from openai import OpenAI
from ..gameplay.prompt import get_system_prompt
from ..utils.global_logger import get_vlm_logger


# ─────────────────────────────────────────────
# Lazy import for HF + LLaVA-OneVision support
# ─────────────────────────────────────────────
def _lazy_import_hf():
    """
    Lazily import the HuggingFace and LLaVA-OneVision dependencies.

    This avoids importing heavy libraries unless we actually need the
    LLaVA-OneVision local path.
    """
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        try:
            # LLaVA-OneVision-specific helper
            from qwen_vl_utils import process_vision_info
        except Exception as e:
            raise ImportError(
                "Missing dependency: qwen_vl_utils. This is usually packaged "
                "together with LLaVA-OneVision. Please install it first."
            ) from e
        return torch, AutoProcessor, AutoModelForCausalLM, process_vision_info
    except Exception as e:
        raise ImportError(
            "Missing HuggingFace dependencies. Please install at least:\n"
            "  pip install transformers torch  (GPU build recommended)"
        ) from e


class BaseModel:
    """
    Unified VLM interface.

    - Default path: OpenAI-compatible Chat Completions API (for OpenAI / third-party endpoints).
    - Special case: if the `model` name contains 'llava-onevision', the calls are routed
      through a local HuggingFace LLaVA-OneVision model.
    - Both paths share the same message construction:
      * a single system prompt from `get_system_prompt()`
      * one user message containing text + up to `max_images` images

    The class returns raw text output without additional parsing or reaction logic.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        *,
        max_tokens: int = 5120,
        temperature: float = 0.0,
        top_p: float = 1.0,
        rate_limit_per_min: Optional[int] = 20,
        supports_vision: Optional[bool] = None,
        reasoning_effort: Optional[str] = "minimal",
        # HF / LLaVA-OneVision specific options
        hf_model_id: Optional[str] = None,          # local directory or remote model ID
        hf_torch_dtype: Optional[str] = "bfloat16", # "auto" | "bfloat16" | "float16" | torch.dtype
        hf_device_map: str = "auto",
        max_images: int = 3,                        # max number of images per call
        hf_gpu: Optional[Union[int, str]] = 3,      # explicit GPU index or "cuda:i"; None = do not force
    ):
        # OpenAI-compatible client (can be OpenAI or any compatible third-party endpoint)
        self.client = OpenAI(base_url=url, api_key=api_key)

        self.model = model or ""
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.rate_limit_per_min = rate_limit_per_min
        self.logger = get_vlm_logger()
        self.reasoning_effort = reasoning_effort

        # By default, enable vision support unless explicitly disabled.
        self.supports_vision = True if supports_vision is None else bool(supports_vision)

        # Trigger condition for LLaVA-OneVision local inference
        m = self.model.lower()
        self._use_llava = ("llava-onevision" in m) or ("llava onevision" in m)

        # HF configuration and lazy-loaded handles
        # Prefer a local directory if available; otherwise fall back to a remote model ID.
        _default_local = "/data/lingjun/models/llava-ov-delivery-1-epoch-merged"
        _remote_id = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
        if hf_model_id is None:
            self._hf_model_id = _default_local if os.path.isdir(_default_local) else _remote_id
        else:
            self._hf_model_id = hf_model_id

        self._hf_torch_dtype = hf_torch_dtype
        self._hf_device_map = hf_device_map
        self._hf_model = None
        self._hf_processor = None
        self._hf_gpu = self._parse_gpu(hf_gpu)  # -> None | int | "cuda:i"
        self.max_images = int(max_images)

    # ======================================================
    # Public generation entrypoint
    # ======================================================
    def generate(
        self,
        user_prompt: str,
        images: Optional[Union[Any, List[Any]]] = None,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: int = 1,
        retry: int = 3,
        rate_limit_per_min: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a text response given:
        - user_prompt: textual user query or instruction.
        - images: optional image or list of images (up to `max_images`), supporting:
          * numpy arrays
          * raw bytes
          * PIL Image objects
          * URLs or paths given as strings

        Returns:
        - model output text (str), stripped of leading/trailing whitespace.
        """
        max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)
        temperature = self.temperature if temperature is None else float(temperature)
        top_p = self.top_p if top_p is None else float(top_p)
        _ = self.rate_limit_per_min if rate_limit_per_min is None else rate_limit_per_min  # reserved

        messages = self._build_messages(user_prompt, images)

        if self._use_llava:
            # Route through local LLaVA-OneVision (HF) pipeline
            return self._generate_llava_onevision(messages, max_tokens, temperature, top_p)
        else:
            # Default: OpenAI-compatible Chat Completions API
            return self._generate_openai_compatible(
                messages, max_tokens, temperature, top_p, n, retry, reasoning_effort, **kwargs
            )

    # ======================================================
    # Shared message construction for both backends
    # ======================================================
    def _build_messages(
        self,
        user_prompt: str,
        images: Optional[Union[Any, List[Any]]],
    ) -> List[dict]:
        """
        Build a standard chat-style message list:

        [
          {"role": "system", "content": <system prompt>},
          {"role": "user",   "content": [<image/text parts>]}
        ]
        """
        messages: List[dict] = [{"role": "system", "content": get_system_prompt()}]

        user_content: List[dict] = []
        if images:
            if not isinstance(images, (list, tuple)):
                images = [images]
            for img in list(images)[: self.max_images]:
                if img is None:
                    continue
                if not self.supports_vision:
                    # If the current model is used in non-vision mode,
                    # silently drop images.
                    continue
                user_content.append(self._to_image_part(img))

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    # ======================================================
    # OpenAI-compatible chat.completions path
    # ======================================================
    def _generate_openai_compatible(
        self,
        messages: List[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        n: int,
        retry: int,
        reasoning_effort: Optional[str],
        **kwargs,
    ) -> str:
        """
        Call an OpenAI-compatible chat.completions endpoint with the assembled messages.
        Handles optional `reasoning_effort` for supported models.
        """
        # Attach ephemeral cache control for non-Mistral models if not already set.
        if "mistral" not in (self.model or "").lower():
            if messages and messages[0].get("role") == "system":
                sysc = messages[0]
                if "cache_control" not in sysc:
                    sysc["cache_control"] = {"type": "ephemeral"}

        # Optional reasoning_effort (available only for selected models).
        effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        effort_set = {"minimal", "low", "medium", "high"}
        use_effort = (
            effort in effort_set
            and ("gpt-5" in (self.model or "").lower())
            and ("gpt-5-chat-latest" not in (self.model or "").lower())
        )

        last_err = None
        for attempt in range(1, retry + 1):
            try:
                create_kwargs = dict(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                )
                if use_effort:
                    create_kwargs["reasoning_effort"] = effort

                create_kwargs.update(kwargs)

                resp = self.client.chat.completions.create(**create_kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                self.logger.error(f"[BaseModel] OpenAI path attempt {attempt} failed: {e}")
                last_err = e

        raise RuntimeError(f"VLM generate failed after {retry} tries: {last_err}")

    # ======================================================
    # HF / LLaVA-OneVision path
    # ======================================================
    def _init_llava_if_needed(self):
        """
        Lazily load the LLaVA-OneVision model and processor from HuggingFace.

        The model is loaded either from a local directory (offline mode) or
        from a remote model hub ID, depending on `self._hf_model_id`.
        """
        if self._hf_model is not None and self._hf_processor is not None:
            return

        torch, AutoProcessor, AutoModelForCausalLM, _ = _lazy_import_hf()

        # Resolve dtype
        dtype = self._hf_torch_dtype or "auto"
        if isinstance(dtype, str):
            s = dtype.lower()
            if s == "auto":
                torch_dtype = "auto"
            elif s in ("bf16", "bfloat16"):
                torch_dtype = torch.bfloat16
            elif s in ("fp16", "float16", "half"):
                torch_dtype = torch.float16
            else:
                torch_dtype = "auto"
        else:
            torch_dtype = dtype

        # Decide if we are loading from a local directory (offline) or remote hub
        local_dir = os.path.isdir(self._hf_model_id)

        # Device map
        device_map = self._hf_device_map
        if self._hf_gpu is not None:
            gpu_index = self._hf_gpu
            if isinstance(gpu_index, str) and gpu_index.startswith("cuda:"):
                try:
                    gpu_index = int(gpu_index.split(":")[1])
                except Exception:
                    gpu_index = 0
            device_map = {"": int(gpu_index)}
            try:
                torch.cuda.set_device(int(gpu_index))
            except Exception:
                pass

        # Load model
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self._hf_model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=local_dir,  # if local dir is present, stay fully offline
        )
        # Load processor
        self._hf_processor = AutoProcessor.from_pretrained(
            self._hf_model_id,
            trust_remote_code=True,
            local_files_only=local_dir,
        )

    def _generate_llava_onevision(
        self,
        messages: List[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """
        Run inference using a local HuggingFace LLaVA-OneVision model.

        The function takes the standard `messages` structure, normalizes
        image parts into the format required by LLaVA-OneVision, constructs
        the chat template, and then calls `generate(...)`.
        """
        self._init_llava_if_needed()
        torch, _, _, process_vision_info = _lazy_import_hf()

        # Normalize "image_url" content into the expected {type: "image", image:<PIL|URL>}
        norm_messages: List[dict] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user" and isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        new_content.append(
                            {"type": "image", "image": self._image_url_to_hf_image(url)}
                        )
                    else:
                        new_content.append(part)
                norm_messages.append({"role": role, "content": new_content})
            else:
                norm_messages.append(msg)

        # Convert messages to a single text prompt via chat template
        text = self._hf_processor.apply_chat_template(
            norm_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(norm_messages)

        inputs = self._hf_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Resolve device: prefer explicit `hf_gpu` if provided, otherwise model's device
        if self._hf_gpu is not None:
            dev = (
                str(self._hf_gpu)
                if isinstance(self._hf_gpu, str)
                else f"cuda:{int(self._hf_gpu)}"
            )
            device = torch.device(dev)
        else:
            device = next(self._hf_model.parameters()).device

        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        do_sample = temperature > 1e-6
        with torch.inference_mode():
            gen_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=(temperature if do_sample else None),
                top_p=(top_p if do_sample else None),
            )
            # Strip the input prompt tokens to keep only the newly generated text.
            trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], gen_ids)]
            out_text = self._hf_processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        return (out_text[0] if out_text else "").strip()

    # ======================================================
    # GPU argument parsing helper
    # ======================================================
    def _parse_gpu(self, gpu: Optional[Union[int, str]]) -> Optional[Union[int, str]]:
        """
        Normalize the `hf_gpu` argument into one of:
        - None
        - an integer GPU index
        - a string "cuda:i"
        """
        if gpu is None:
            return None
        if isinstance(gpu, int):
            return max(0, gpu)

        s = str(gpu).strip().lower()
        if s.isdigit():
            return int(s)
        if s.startswith("cuda:"):
            tail = s.split(":", 1)[1]
            return int(tail) if tail.isdigit() else s
        return None

    # ======================================================
    # Image conversions for OpenAI-style image parts
    # ======================================================
    def _to_image_part(self, image_obj: Any) -> dict:
        """
        Convert various image input types into a valid
        {"type": "image_url", "image_url": {"url": ...}} structure.

        Supported:
        - numpy arrays
        - raw bytes
        - URLs or file paths given as strings
        - PIL Image objects
        - Other types: fallback to string conversion
        """
        # numpy ndarray
        if isinstance(image_obj, np.ndarray):
            b64 = self._ndarray_to_b64jpeg(image_obj)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        # raw bytes
        if isinstance(image_obj, (bytes, bytearray)):
            b64 = base64.b64encode(image_obj).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        # string: treat as URL or file path or raw base64 payload
        if isinstance(image_obj, str):
            s = image_obj.strip()
            if s.startswith(("http://", "https://", "data:image")):
                return {"type": "image_url", "image_url": {"url": s}}

            if os.path.isfile(s):
                with open(s, "rb") as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode("utf-8")
                return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

            # Fallback: assume `s` is already a base64 string
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{s}"}}

        # PIL Image
        if isinstance(image_obj, Image.Image):
            b64 = self._pil_to_b64jpeg(image_obj)
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        # Fallback: convert to string
        return {"type": "image_url", "image_url": {"url": str(image_obj)}}

    def _image_url_to_hf_image(self, url: str) -> Union[str, Image.Image]:
        """
        Convert an `image_url` field from the OpenAI-style message into an object
        consumable by LLaVA-OneVision:
        - http(s) URL → kept as string (LLaVA can fetch it).
        - data URL or bare base64 → decoded into a PIL Image.
        """
        try:
            s = (url or "").strip()
            # HTTP/HTTPS URL: keep as URL, HF pipeline will handle it.
            if s.startswith(("http://", "https://")):
                return s

            # Data URL with embedded base64
            if s.startswith("data:image"):
                head, b64 = s.split(",", 1)
                raw = base64.b64decode(b64)
                return Image.open(io.BytesIO(raw)).convert("RGB")

            # Assume bare base64 payload
            raw = base64.b64decode(s)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            # Fallback: return a tiny white image to avoid crashing the pipeline.
            return Image.new("RGB", (1, 1), color=(255, 255, 255))

    # ======================================================
    # ndarray / PIL to base64 JPEG helpers
    # ======================================================
    @staticmethod
    def _ndarray_to_b64jpeg(arr: np.ndarray, quality: int = 90) -> str:
        """
        Encode a numpy array as a base64 JPEG.

        Supported shapes:
        - (H, W)       → grayscale
        - (H, W, 3)    → RGB
        - (H, W, 4)    → RGBA
        """
        if arr.ndim == 2:
            mode = "L"
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = "RGB"
        elif arr.ndim == 3 and arr.shape[2] == 4:
            mode = "RGBA"
        else:
            raise ValueError(f"Unsupported ndarray shape: {arr.shape}")

        img = Image.fromarray(arr.astype(np.uint8), mode=mode)
        return BaseModel._pil_to_b64jpeg(img, quality=quality)

    @staticmethod
    def _pil_to_b64jpeg(img: Image.Image, quality: int = 90) -> str:
        """
        Encode a PIL Image as a base64 JPEG.

        Non-RGB/L images are converted to RGB first.
        """
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")