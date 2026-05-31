# -*- coding: utf-8 -*-
"""GUI analysis backends: local GUI-Owl model and cloud API."""
import base64
import json
import os
import re
import sys
import threading
from typing import Callable, Optional

import requests

from config import (
    ANALYSIS_PROMPT,
    DASHSCOPE_BASE_URL,
    DASHSCOPE_MODEL,
    LOCAL_GUI_OWL_MODEL_ID,
    LOCAL_MODEL_CACHE_DIRNAME,
    LOCAL_MODEL_CPU_MEMORY,
    LOCAL_MODEL_GPU_MEMORY,
    LOCAL_MODEL_MAX_NEW_TOKENS,
    LOCAL_IMAGE_MAX_EDGE,
)

_LOCAL_ENGINE = None
_LOCAL_ENGINE_LOCK = threading.Lock()
_NULL_STREAM = None


def _image_to_base64_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


class LocalGUIOwlEngine:
    """Local model runtime for GUI-Owl 1.5 4B."""

    def __init__(self, model_id: str, base_dir: str):
        self.model_id = model_id
        safe_model_dir = model_id.replace("/", "__")
        self.model_dir = os.path.join(base_dir, LOCAL_MODEL_CACHE_DIRNAME, safe_model_dir)
        self._model = None
        self._processor = None
        self._cpu_fallback = False

    def _is_model_deployed(self) -> bool:
        required = (
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        )
        return all(os.path.isfile(os.path.join(self.model_dir, name)) for name in required)

    def ensure_downloaded(
        self,
        progress_cb: Optional[Callable[[str], None]] = None,
        allow_download: bool = True,
    ):
        _ensure_safe_stdio()
        if self._is_model_deployed():
            return
        if not allow_download:
            raise RuntimeError(
                "本地模型未部署完成。请先点击「一键部署本地模型」，并保持网络可访问 Hugging Face。"
            )
        if progress_cb:
            progress_cb("首次部署：正在下载本地 GUI-Owl 模型（体积较大，请耐心等待）...")
        os.makedirs(self.model_dir, exist_ok=True)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.model_id,
            local_dir=self.model_dir,
            token=os.environ.get("HF_TOKEN") or None,
        )
        if not self._is_model_deployed():
            raise RuntimeError("模型下载未完成，请检查网络后重试。")
        if progress_cb:
            progress_cb("模型下载完成。")

    def ensure_loaded(
        self,
        progress_cb: Optional[Callable[[str], None]] = None,
        allow_download: bool = True,
    ):
        _ensure_safe_stdio()
        if self._model is not None and self._processor is not None:
            return
        self.ensure_downloaded(progress_cb=progress_cb, allow_download=allow_download)
        if progress_cb:
            progress_cb("正在加载本地 GUI-Owl 模型到内存/显存...")

        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        dtype = torch.float16 if (torch.cuda.is_available() and not self._cpu_fallback) else torch.float32
        if torch.cuda.is_available() and not self._cpu_fallback:
            max_memory = {0: LOCAL_MODEL_GPU_MEMORY, "cpu": LOCAL_MODEL_CPU_MEMORY}
            device_map = "auto"
        else:
            max_memory = None
            device_map = "cpu"

        self._processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            local_files_only=True,
        )
        self._model.eval()
        if progress_cb:
            mode = "CPU 兜底模式" if self._cpu_fallback else "GPU/CPU 混合模式"
            progress_cb(f"本地模型已就绪（{mode}）。")

    def _reload_cpu_fallback(self):
        self._model = None
        self._processor = None
        self._cpu_fallback = True
        self.ensure_loaded(progress_cb=None, allow_download=False)

    def analyze(self, image_path: str, timestamp_sec: float = 0.0) -> dict:
        from PIL import Image
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("Local model is not loaded")

        prompt_with_ts = (
            f"This screenshot is captured at video timestamp {timestamp_sec:.1f}s.\n\n"
            + ANALYSIS_PROMPT
        )
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > LOCAL_IMAGE_MAX_EDGE:
            image.thumbnail((LOCAL_IMAGE_MAX_EDGE, LOCAL_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_with_ts},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        try:
            with torch.inference_mode():
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=LOCAL_MODEL_MAX_NEW_TOKENS,
                )
        except RuntimeError as e:
            msg = str(e).lower()
            if "cuda out of memory" not in msg:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._reload_cpu_fallback()
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)
            with torch.inference_mode():
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=min(LOCAL_MODEL_MAX_NEW_TOKENS, 192),
                )
        finally:
            if torch.cuda.is_available() and not self._cpu_fallback:
                torch.cuda.empty_cache()
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, out_ids)
        ]
        content = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return _parse_response(content, timestamp_sec)


def _ensure_safe_stdio():
    """PyInstaller windowed app may have None stdout/stderr; patch them for libs."""
    global _NULL_STREAM
    if _NULL_STREAM is None:
        _NULL_STREAM = open(os.devnull, "w", encoding="utf-8")
    if getattr(sys, "stdout", None) is None:
        sys.stdout = _NULL_STREAM
    if getattr(sys, "stderr", None) is None:
        sys.stderr = _NULL_STREAM


def _get_or_create_local_engine(
    base_dir: str,
    model_id: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
    allow_download: bool = True,
):
    global _LOCAL_ENGINE
    with _LOCAL_ENGINE_LOCK:
        target_model_id = model_id or LOCAL_GUI_OWL_MODEL_ID
        if _LOCAL_ENGINE is None or _LOCAL_ENGINE.model_id != target_model_id:
            _LOCAL_ENGINE = LocalGUIOwlEngine(target_model_id, base_dir=base_dir)
        _LOCAL_ENGINE.ensure_loaded(
            progress_cb=progress_cb,
            allow_download=allow_download,
        )
        return _LOCAL_ENGINE


def deploy_local_model(
    base_dir: str,
    model_id: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Download and warm up local model for first-time deployment."""
    engine = _get_or_create_local_engine(
        base_dir=base_dir,
        model_id=model_id,
        progress_cb=progress_cb,
        allow_download=True,
    )
    return {
        "ok": True,
        "model_id": engine.model_id,
        "model_dir": engine.model_dir,
    }


def ensure_local_model_ready(
    base_dir: str,
    model_id: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
    allow_download: bool = False,
):
    _get_or_create_local_engine(
        base_dir=base_dir,
        model_id=model_id,
        progress_cb=progress_cb,
        allow_download=allow_download,
    )


def analyze_screenshot(
    image_path: str,
    api_key: str,
    timestamp_sec: float = 0.0,
    backend: str = "local",
    app_base_dir: str = "",
    base_url: str = "",
    model: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Analyze screenshot with local backend or cloud API backend.

    Returns a dict with keys: app_name, page_name, elements, visible_text,
    user_action, description.  On failure returns a dict with an "error" key.
    """
    backend = (backend or "local").lower()
    if backend == "local":
        try:
            if not app_base_dir:
                app_base_dir = os.getcwd()
            engine = _get_or_create_local_engine(
                base_dir=app_base_dir,
                model_id=model,
                progress_cb=progress_cb,
                allow_download=False,
            )
            return engine.analyze(image_path=image_path, timestamp_sec=timestamp_sec)
        except Exception as e:
            return {"error": str(e)}

    base_url = base_url or DASHSCOPE_BASE_URL
    model = model or DASHSCOPE_MODEL
    url = f"{base_url}/chat/completions"

    image_b64 = _image_to_base64_url(image_path)
    prompt_with_ts = (
        f"This screenshot is captured at video timestamp {timestamp_sec:.1f}s.\n\n"
        + ANALYSIS_PROMPT
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_with_ts},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            }
        ],
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_response(content, timestamp_sec)
    except requests.exceptions.HTTPError as e:
        error_body = ""
        try:
            error_body = e.response.text
        except Exception:
            pass
        return {"error": f"HTTP {e.response.status_code}: {error_body}"}
    except Exception as e:
        return {"error": str(e)}


def _parse_response(raw_text: str, timestamp_sec: float) -> dict:
    """Extract JSON from the model's response text."""
    cleaned = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)

    if not cleaned.startswith("{"):
        brace_start = cleaned.find("{")
        if brace_start != -1:
            brace_end = cleaned.rfind("}") + 1
            cleaned = cleaned[brace_start:brace_end]

    default = {
        "timestamp_sec": timestamp_sec,
        "app_name": "",
        "page_name": "",
        "elements": [],
        "visible_text": [],
        "user_action": "",
        "description": "",
    }

    try:
        parsed = json.loads(cleaned)
        result = {**default, **parsed}
        result["timestamp_sec"] = timestamp_sec
        return result
    except json.JSONDecodeError:
        default["description"] = raw_text[:500]
        default["_raw"] = raw_text
        return default
