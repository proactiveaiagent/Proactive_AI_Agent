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

TRANSLATE_PROMPT = (
    "Translate the following text into English. "
    "Output ONLY the English translation, no explanations, no quotation marks.\n\n{text}"
)


def make_translator(qwen_api_url: str = "http://localhost:8000",
                    timeout: int = 30) -> callable:
    """返回带缓存（进程内 dict）的翻译函数 translate(text) -> str。

    相同文本只翻译一次（缓存命中免 LLM 调用）。LLM 调用失败抛异常，
    由调用方（memory.add / _query_tokens）捕获后降级。
    """
    cache: dict = {}

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
        return en

    return translate
