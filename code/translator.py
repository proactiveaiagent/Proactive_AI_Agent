"""
translator.py — 非英文 → 英文翻译（统一英文分词策略的翻译环节）
=================================================================
设计依据：设计说明书 docs/memory-model-design.md §6.2 / §9.3——
「统一英文分词：非英文语种先翻译为英文再分词。写入时预翻译存 normalized，
查询词同样翻译；翻译走 LLM 时离线预翻译并缓存，检索同步链路上不得实时调用 LLM。」

本模块提供带进程内缓存的翻译函数，供记忆模块写入路径（add/update 预翻译）
与查询路径（查询词翻译）注入使用。翻译失败由调用方降级（原文兜底，不崩溃）。

用法：
    from translator import make_translator
    translate = make_translator(qwen_api_url="http://localhost:8000")
    en = translate("在图书馆学习")   # -> "studying in the library"（命中缓存后免 LLM）
"""

import requests
import json
import re
import os
from pathlib import Path

TRANSLATE_PROMPT = (
    "Translate the following text into English. "
    "Output ONLY the English translation, no explanations, no quotation marks.\n\n{text}"
)


def make_translator(qwen_api_url: str = "http://127.0.0.1:8001",
                    timeout: int = 30, cache_path=None) -> callable:
    """返回带缓存（进程内 dict）的翻译函数 translate(text) -> str。

    相同文本只翻译一次（缓存命中免 LLM 调用）。LLM 调用失败抛异常，
    由调用方（memory.add / _query_tokens）捕获后降级。
    """
    cache: dict = {}
    path = Path(cache_path) if cache_path else None
    if path and path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                cache.update({str(k): str(v) for k, v in saved.items()})
        except (OSError, ValueError, TypeError):
            pass

    def translate(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text
        if text in cache:
            return cache[text]

        resp = requests.post(
            f"{qwen_api_url.rstrip('/')}/consolidate",
            json={"prompt": TRANSLATE_PROMPT.format(text=text)},
            timeout=timeout,
        )
        resp.raise_for_status()
        en = (resp.json().get("analysis") or "").strip().strip('"').strip("'")
        if not en:
            raise ValueError(f"empty translation for {text!r}")
        cache[text] = en
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
                temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, path)
            except OSError:
                pass
        return en

    return translate


_CJK = re.compile(r"[\u3400-\u9fff]")
_FIELD_KEYS = ("scene", "user_action", "interaction_details", "location",
               "activity", "time_occasion")


def translate_structured_fields(fields, person_observations, temporal_facts, translate):
    """Translate descriptive structured fields; keep names and source evidence intact."""
    def tr(value):
        if not isinstance(value, str) or not _CJK.search(value):
            return value
        try:
            translated = translate(value)
            return translated if isinstance(translated, str) and translated.strip() else value
        except Exception:
            return value

    for key in _FIELD_KEYS:
        fields[key] = tr(fields.get(key))
    for key in ("objects", "environments"):
        value = fields.get(key)
        if isinstance(value, list):
            fields[key] = [tr(item) if isinstance(item, str) else item for item in value]
    if isinstance(person_observations, list):
        for row in person_observations:
            if not isinstance(row, dict):
                continue
            for key in ("appearance", "face_features", "profile_context"):
                row[key] = tr(row.get(key))
            relations = row.get("relationships")
            if isinstance(relations, list):
                row["relationships"] = [tr(item) if isinstance(item, str) else item for item in relations]
    if isinstance(temporal_facts, list):
        for row in temporal_facts:
            if not isinstance(row, dict):
                continue
            for key in ("predicate", "value", "from_state", "to_state", "from_location", "to_location"):
                original = row.get(key)
                if key == "value" and isinstance(original, str) and _CJK.search(original):
                    row.setdefault("source_value", original)
                row[key] = tr(original)
    return fields, person_observations, temporal_facts
