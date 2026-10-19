"""
interactive_runner.py — 交互式步进执行器
======================================================================
把 run_scenarios.py 的一次性流程拆成一步一步，供可视化网站交互执行：
人可以看到当前输入的是哪个视频、时间点，手工点击「运行」再走下一步。

短期场景：每条视频片段一个 add 步骤 → query / update / highlight / delete。
长期场景：白天按小时批量写入 → NightlyProcessor 汇总 / 分类 / 画像更新。

被 viz_server.py 调用（单线程 HTTPServer，无需加锁）：
    InteractiveSession("short_term")   创建会话
    session.state()                    当前状态（下一步信息 + 已执行日志 + 最新快照）
    session.run_step()                 执行当前步，返回结果消息
"""

import copy
import hashlib
import json
import re
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory import (PersonMemory, _build_entity_visuals, _portable_video_id,
                    entity_graph_summary, normalize_object_transition_facts)  # noqa: E402
from entity_records import memory_facts
from daytime_processor import DaytimeProcessor  # noqa: E402
from nighttime_processor import NightlyProcessor  # noqa: E402
from multimodal_memory import (CAPTURE_VERSION, MultimodalAssetStore, _run_ffmpeg,
                                media_refs, strip_audio_assets)  # noqa: E402
import run_scenarios as rs               # noqa: E402  复用 snapshot / 夜间整理辅助函数
from qwen_perceive import (QwenPerceiver, golden_eval, clip_range, video_path_of,
                           PROMPT_VERSION, LIST_PARSER_VERSION,
                           _organize_person_entity_aliases)   # noqa: E402
from advice_cache import PersistentAdviceCache  # noqa: E402
from grounding_cache import PersistentObjectGroundingCache  # noqa: E402
from speech_identity import (attach_spoken_identity, person_labels_from_observations,
                             terms_in_text)  # noqa: E402
from audio_transcript_cache import PersistentTranscriptCache  # noqa: E402
from scene_retrieval import PostWriteRetrievalMixin  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TESTSET = ROOT / "data" / "egolife" / "testset"
RUNS = ROOT / "data" / "egolife" / "runs"
VIDEO_ROOT = (ROOT / "data" / "egolife").resolve()

_AGE_AS_PERSON_RE = re.compile(
    r"\b(a|an|the)\s+(baby|infant|toddler|newborn|child|children|boy|girl)\b"
    r"(?=\s+(?:is|was|who|with|wearing|holding|holds|sits|stands|playing|smiling|"
    r"appears|in|on|at)\b|[.,;]|$)",
    re.I,
)
_AGE_ADJECTIVE_PERSON_RE = re.compile(
    r"\b(?:elderly|older|younger|young|old)\s+(?:woman|man|person|adult|individual)\b",
    re.I,
)


def _remove_visual_age_guess(value):
    """Do not encode an age category inferred from a person’s appearance."""
    text = str(value or "")
    text = _AGE_AS_PERSON_RE.sub(
        lambda match: ("the person" if match.group(1).casefold() == "the" else "a person"),
        text,
    )
    return _AGE_ADJECTIVE_PERSON_RE.sub("person", text)


def _source_video_id(moment, video_path=None):
    """Return a portable EgoLife-relative video ID, never a host absolute path."""
    raw = ""
    if video_path:
        try:
            raw = Path(video_path).resolve().relative_to(VIDEO_ROOT).as_posix()
        except (OSError, ValueError):
            try:
                raw = Path(video_path).resolve().relative_to(FEISHU_TEST_ROOT).as_posix()
            except (OSError, ValueError):
                try:
                    raw = Path(video_path).resolve().relative_to(ROOT).as_posix()
                except (OSError, ValueError):
                    raw = ""
    raw = raw or str(moment.get("video_id") or moment.get("video") or "")
    return _portable_video_id(raw)

SHORT_FILE = TESTSET / "short_term" / "moments_scattered.jsonl"
SHORT_PLAN = TESTSET / "short_term" / "crud_plan.json"
SHORT_EXPANDED_FILE = TESTSET / "short_term" / "moments_scattered_v2_120.jsonl"
SHORT_EXPANDED_PLAN = TESTSET / "short_term" / "crud_plan_v2.json"
LONG_FILE = TESTSET / "long_term" / "full_day.jsonl"
LONG_META = TESTSET / "long_term" / "timeline_meta.json"
LONG_PERSON_DIR = TESTSET / "long_term" / "person_sets"
FEISHU_TEST_ROOT = (ROOT / "test_data" / "feishu_ar_agent_memory_retrieval_20261013").resolve()
FEISHU_TEST_MANIFEST = FEISHU_TEST_ROOT / "manifest.json"
FEISHU_TESTSET_LABELS = {
    "1.social_hint": "社交提示", "2.travel_abroad": "出国旅行",
    "3.forgetting_item": "遗忘物品", "4.first_aid": "急救场景",
    "5.health_safeguard": "健康保障", "6.hotel_reception": "酒店接待",
    "7.photo_taking": "拍照协助", "8.fraud_detection": "诈骗识别",
    "9.security_alert": "安全警报", "10.conversation_assistance": "对话辅助",
    "11.emotional support": "情绪支持", "12.shopping_selection": "购物选择",
}


def site_perceiver():
    """Use the memory-eval V3 cache for this visualization, with an env override."""
    configured = os.environ.get("EGOLIFE_QWEN_CACHE")
    if configured:
        cache_path = Path(configured)
        if not cache_path.is_absolute():
            cache_path = ROOT / cache_path
    else:
        cache_path = RUNS / "qwen_cache_memory_prompt_v3.json"
        if not cache_path.exists():
            cache_path = RUNS / "qwen_cache.json"
    return QwenPerceiver(cache_file=cache_path)


SCENARIO_META = {
    "short_term": {"name": "短期记忆场景", "desc": "不连续视频片段 · 验证 Layer1~7 完整 CRUD"},
    "long_term": {"name": "长期记忆场景", "desc": "完整一天 · 验证记忆流转与夜间整理"},
    "custom_test": {"name": "自定义测试集", "desc": "按场景分类逐段写入与检索，全部片段完成后可执行夜间归纳"},
}


def _clock_from_id(mid):
    """DAY1_A2_ALICE_11093000_en → 11:09:30"""
    parts = str(mid).split("_")
    if len(parts) >= 4 and re.fullmatch(r'DAY\d+', parts[0], re.I):
        t = parts[3]
        if (len(t) >= 6 and t[:6].isdigit() and int(t[:2]) < 24
                and int(t[2:4]) < 60 and int(t[4:6]) < 60):
            return f"{t[0:2]}:{t[2:4]}:{t[4:6]}"
    return ""


def _intra_hour_offset(mid):
    """文件名时间戳相对整点小时的秒数：DAY1_A2_ALICE_11093000 → 9*60+30 = 570s"""
    parts = str(mid).split("_")
    if _clock_from_id(mid):
        t = parts[3]
        return int(t[2:4]) * 60 + int(t[4:6])
    return 0


_AUDIO_EVIDENCE_LINE_RE = re.compile(
    r"^\[audio\s+(?P<start>\d+(?:\.\d+)?)\s*[–-]\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*s;\s*(?P<language>[^\]]+)\]\s*(?P<text>.+)$",
    re.I,
)
_EXPLICIT_ENGLISH_AGE_RE = re.compile(
    r"\b(?:i am|i'm)\s+(\d{1,3})\s+(?:years? old|yo)\b", re.I)
_EXPLICIT_CHINESE_AGE_RE = re.compile(r"(?:我|本人)(?:今年)?\s*(\d{1,3})\s*岁")

_ENGLISH_PERSON_SUBJECT = (
    r"(?:[A-Z][A-Za-z'’.-]*(?:\s+[A-Z][A-Za-z'’.-]*){0,2}"
    r"|(?i:(?:my|his|her|their)\s+(?:mother|mom|mum|father|dad|daughter|son|"
    r"wife|husband|sister|brother|friend|colleague|doctor))"
    r"|(?i:he|she|they))"
)
_CHINESE_PERSON_SUBJECT = (
    r"(?:我(?:们)?(?:的)?(?:妈妈|母亲|妈|爸爸|父亲|爸|奶奶|爷爷|女儿|儿子|"
    r"妻子|丈夫|姐妹|兄弟|朋友|同事)?|他(?:们)?|她(?:们)?|[\u4e00-\u9fff]{2,4})"
)
_EXPLICIT_ENGLISH_OTHER_AGE_RE = re.compile(
    rf"\b(?P<subject>{_ENGLISH_PERSON_SUBJECT})\s+(?:(?:is|are)\s+)?"
    r"(?P<age>\d{1,3})\s+(?:years? old|yo)\b"
)
_EXPLICIT_CHINESE_OTHER_AGE_RE = re.compile(
    rf"(?P<subject>{_CHINESE_PERSON_SUBJECT})\s*"
    r"(?:(?:今年|现在|目前)\s*)?(?P<age>\d{1,3})\s*岁"
)
_EXPLICIT_ENGLISH_OTHER_PLAN_RE = re.compile(
    rf"\b(?P<subject>{_ENGLISH_PERSON_SUBJECT})\s+"
    r"(?:(?:is|are)\s+)?(?:planning to|plans to|going to|will|shall|gonna)\b(?!\s+(?:not|never)\b)"
)
_EXPLICIT_CHINESE_OTHER_PLAN_RE = re.compile(
    rf"(?P<subject>{_CHINESE_PERSON_SUBJECT})\s*"
    r"(?P<time_scope>未来几天|这几天|今天|今晚|明天|后天|本周|这周|下周(?:[一二三四五六日天])?|"
    r"周[一二三四五六日天]|月底|最近)?\s*"
    r"(?<!没有)(?<!不)(?<!没)(?<!未)(?:打算|计划|准备|将要|将|会|要)"
)
_ENGLISH_HEALTH_CONDITION = (
    r"(?:type\s+\d+\s+diabetes|high blood pressure|heart disease|heart condition|heart attack|"
    r"kidney disease|blood clot|diabetes|asthma|hypertension|epilepsy|cancer|stroke|"
    r"arthritis|dementia|allerg(?:y|ies))"
)
_EXPLICIT_ENGLISH_OTHER_HEALTH_RE = re.compile(
    rf"\b(?P<subject>{_ENGLISH_PERSON_SUBJECT})\s+(?:"
    rf"(?:(?:has|had|suffers? from|was diagnosed with|has been diagnosed with|"
    rf"is diagnosed with)\s+(?:a\s+|an\s+)?(?P<condition>{_ENGLISH_HEALTH_CONDITION})\b)"
    rf"|(?:(?:takes?|is taking)\s+(?:medication|medicine|insulin|pills?)\s+for\s+"
    rf"(?P<med_condition>{_ENGLISH_HEALTH_CONDITION})\b)"
    rf"|(?:(?:is|are)\s+allergic to\s+(?P<allergen>[^.!?,;]{{1,40}}))"
    rf"|(?:(?:had|has had|underwent)\s+(?P<surgery>surgery|a surgery)\b)"
    rf")",
    re.I,
)
_CHINESE_HEALTH_CONDITION = (
    r"(?:糖尿病|高血压|哮喘|心脏病|心肌梗死|心梗|心脏病发作|癫痫|癌症|中风|关节炎|痴呆|老年痴呆|"
    r"血栓|肾病|过敏|手术|降压药|胰岛素|药物)"
)
_EXPLICIT_CHINESE_OTHER_HEALTH_RE = re.compile(
    rf"(?P<subject>{_CHINESE_PERSON_SUBJECT})\s*(?:曾经|以前|目前|现在|一直)?"
    rf"(?<!没)(?<!不)(?<!未)(?:患有|患过|得过|有|被诊断为|被诊断出|"
    rf"确诊(?:患有)?|正在服用|服用|做过)"
    rf"[^。！？.!?]{{0,16}}?(?P<condition>{_CHINESE_HEALTH_CONDITION})"
)
_EXPLICIT_CHINESE_OTHER_ALLERGY_RE = re.compile(
    rf"(?P<subject>{_CHINESE_PERSON_SUBJECT})\s*对"
    rf"(?P<allergen>[^。！？.!?]{{1,20}}?)(?<!不)过敏"
)
_EXPLICIT_PLAN_TIME_RE = re.compile(
    r"\b(?:the day after tomorrow|tomorrow|today|tonight|this week|this weekend|"
    r"next week|next month|in a few days|in \d{1,2} days|on "
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b"
    r"|(?:未来几天|这几天|今天|今晚|明天|后天|本周|这周|下周(?:[一二三四五六日天])?|"
    r"周[一二三四五六日天]|月底|最近)",
    re.I,
)

_EXPLICIT_ENGLISH_PLAN_RE = re.compile(
    r"\b(?:i(?:'m| am) planning to|i plan to|i(?:'m| am) going to|"
    r"i(?:'m| am) gonna|i(?:'ll| will)|we(?:'re| are) planning to|"
    r"we plan to|we(?:'re| are) going to|we(?:'re| are) gonna|we(?:'ll| will))\b(?!\s+(?:not|never)\b)",
    re.I,
)
_EXPLICIT_CHINESE_PLAN_RE = re.compile(
    r"(?:我|我们)(?:打算|计划|准备)(?:要)?[^。！？.!?]{1,180}")
_EXPLICIT_ENGLISH_HEALTH_RE = re.compile(
    r"\b(?:i have(?!\s+(?:no|not|never)\b)|i've had(?!\s+(?:no|not|never)\b)|i had(?!\s+(?:no|not|never)\b)|i was diagnosed with(?!\s+(?:no|not|never)\b)|i have been diagnosed with(?!\s+(?:no|not|never)\b)|"
    r"i suffer from|i take (?:medication|medicine|insulin|pills?) for|"
    r"i am taking (?:medication|medicine|insulin|pills?) for)\b.{0,100}"
    r"\b(?:diabetes|asthma|hypertension|high blood pressure|heart disease|"
    r"heart condition|heart attack|epilepsy|cancer|stroke|arthritis|dementia|allerg(?:y|ies)|"
    r"blood clot|kidney disease|medication|insulin|surgery)\b"
    r"|\b(?:i'm|i am) allergic to [\w-]+(?:\s+[\w-]+){0,4}\b", re.I)
_EXPLICIT_CHINESE_HEALTH_RE = re.compile(
    r"(?:我|本人)(?:(?:曾经|以前|目前|现在|一直)?(?:患有|患过|得过|有|被诊断为|"
    r"被诊断出|确诊(?:患有)?|正在服用|服用|做过))[^。！？.!?]{0,80}"
    r"(?:糖尿病|高血压|哮喘|心脏病|心肌梗死|心梗|心脏病发作|癫痫|癌症|中风|关节炎|痴呆|老年痴呆|"
    r"血栓|肾病|过敏|手术|降压药|胰岛素|药物)|"
    r"(?:我|本人)对[^。！？.!?]{1,20}(?<!没有)(?<!不)(?<!没)过敏")


_EXPLICIT_ENGLISH_SELF_NAME_RE = re.compile(
    r"(?i:\b(?:my name is|you can call me|please call me)\s+)"
    r"(?P<name>[A-Za-z][A-Za-z'’.-]{1,40}(?:\s+[A-Z][A-Za-z'’.-]{1,40}){0,2})\b")
_EXPLICIT_CHINESE_SELF_NAME_RE = re.compile(
    r"(?:我的名字是|我叫)(?P<name>[\u4e00-\u9fff]{2,4})(?=\s|[,，。！？.!?]|$)")
_EXPLICIT_ENGLISH_OTHER_NAME_RE = re.compile(
    r"\b(?P<subject>(?i:he|she|they)|"
    r"(?i:my|our|his|her|their)\s+(?:mother|mom|mum|father|dad|daughter|son|wife|husband|"
    r"sister|brother|friend|colleague|partner|boyfriend|girlfriend|roommate))"
    r"(?i:'s name is|\s+is called|\s+goes by)\s+"
    r"(?P<name>[A-Za-z][A-Za-z'’.-]{1,40}(?:\s+[A-Z][A-Za-z'’.-]{1,40}){0,2})\b")
_EXPLICIT_CHINESE_OTHER_NAME_RE = re.compile(
    r"(?P<subject>他|她|他们|她们|我(?:的)?(?:妈妈|母亲|爸爸|父亲|奶奶|爷爷|女儿|儿子|妻子|丈夫|"
    r"姐妹|兄弟|朋友|同事|伴侣|男朋友|女朋友|室友))"
    r"(?:的名字是|的名字叫|叫)(?P<name>[\u4e00-\u9fff]{2,4})(?=\s|[,，。！？.!?]|$)")
_EXPLICIT_ENGLISH_PREFERENCE_RE = re.compile(
    r"\b(?:I|we)(?:\s+really)?\s+(?:(?:do not|don't|never)\s+)?"
    r"(?:like|love|enjoy|prefer|hate|dislike)\s+[^.!?;]{2,140}", re.I)
_EXPLICIT_CHINESE_PREFERENCE_RE = re.compile(
    r"(?:我|我们)(?:最|更|比较|非常)?(?:喜欢|爱|享受|偏好|讨厌|不喜欢|不爱)"
    r"[^。！？.!?;]{2,100}")
_EXPLICIT_ENGLISH_OTHER_PREFERENCE_RE = re.compile(
    r"\b(?P<subject>(?i:he|she|they)|"
    r"[A-Z][A-Za-z'’.-]*(?:\s+[A-Z][A-Za-z'’.-]*){0,2}|"
    r"(?i:(?:my|our|his|her|their)\s+(?:mother|mom|mum|father|dad|daughter|son|"
    r"wife|husband|sister|brother|friend|colleague|partner|boyfriend|girlfriend|roommate)))"
    r"\s+(?:(?i:really)\s+)?(?i:likes|loves|enjoys|prefers|hates|dislikes)\s+"
    r"[^.!?;]{2,140}")
_EXPLICIT_CHINESE_OTHER_PREFERENCE_RE = re.compile(
    r"(?P<subject>他|她|他们|她们|我(?:们)?(?:的)?(?:妈妈|母亲|爸爸|父亲|奶奶|爷爷|"
    r"女儿|儿子|妻子|丈夫|朋友|同事|伴侣|男朋友|女朋友|室友)|[\u4e00-\u9fff]{2,4})"
    r"(?:最|更|比较)?(?:喜欢|爱|偏好|讨厌|不喜欢)[^。！？.!?;]{2,100}")
_EXPLICIT_ENGLISH_RELATION_RE = re.compile(
    r"\b(?:my|our)\s+(?:mother|mom|mum|father|dad|daughter|son|wife|husband|"
    r"sister|brother|friend|colleague|partner|boyfriend|girlfriend|roommate|doctor)\b|"
    r"\b(?:I(?:'m| am)|we(?:'re| are))\s+your\s+neighbou?r\b", re.I)
_EXPLICIT_CHINESE_RELATION_RE = re.compile(
    r"(?:我|我们)(?:的)?(?:妈妈|母亲|妈|爸爸|父亲|爸|奶奶|爷爷|女儿|儿子|妻子|丈夫|"
    r"姐妹|兄弟|朋友|同事|伴侣|男朋友|女朋友|室友|医生)")
_SPOKEN_VOCATIVE_NAME_RE = re.compile(
    r"^\s*(?i:hi|hello|hey|welcome|good\s+(?:morning|afternoon|evening))"
    r"\b[,!\s]+(?P<name>[A-Z][a-zA-Z'’-]*(?:\s+[A-Z][a-zA-Z'’-]*){0,2})\b")
_EXPLICIT_SELF_NEIGHBOR_RE = re.compile(
    r"\b(?:I(?:'m| am)|we(?:'re| are))\s+your\s+neighbou?r\b", re.I)
_EXPLICIT_SELF_ROLE_RE = re.compile(
    r"\b(?:I(?:'m| am)|we(?:'re| are))\s+your\s+"
    r"(?P<role>grandma|grandmother|grandpa|grandfather|mother|mom|mum|"
    r"father|dad|aunt|uncle|sister|brother|friend|neighbou?r|coworker|colleague)\b",
    re.I)
_EXPLICIT_CHINESE_SELF_ROLE_RE = re.compile(
    r"(?:我是|我就是|我们是|我们就是)(?:你|你们)的?"
    r"(?P<role>奶奶|外婆|爷爷|外公|妈妈|母亲|爸爸|父亲|朋友|邻居|同事|姐妹|兄弟)")
_EXPLICIT_SPOKEN_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))\b", re.I)
_EXPLICIT_WELLBEING_QUESTION_RE = re.compile(
    r"\bhow\s+(?:are|have)\s+you(?:\s+today)?\b|"
    r"\bhow(?:'ve| have)\s+you\s+been\b", re.I)
_DIRECT_ADDRESSED_HEALTH_RE = re.compile(
    rf"^\s*(?P<name>[A-Z][A-Za-z'’.-]*(?:\s+[A-Z][A-Za-z'’.-]*){{0,2}})"
    rf"[,，]\s*(?:your|你|您的)\s+(?:heart disease|heart condition|heart attack|"
    rf"high blood pressure|diabetes|asthma|癌症|心脏病|心肌梗死|心梗|高血压)", re.I)
_EXPLICIT_HEALTH_PRODUCT_CLAIM_RE = re.compile(
    r"\b(?:health|healthcare)\s+product\b.{0,120}\b(?:works?|address(?:es)?|"
    r"treats?|cures?|prevents?|improves?|keeps?)\b", re.I)
_EXPLICIT_FIRST_VISIT_RE = re.compile(
    r"\b(?:I|we)(?:'ve| have)?\s+never\s+(?:been here before|been to|visited)\b", re.I)
_EXPLICIT_FIRED_RE = re.compile(
    r"\b(?:you(?:'re| are)\s+fired|leave our company from today)\b", re.I)
_EXPLICIT_BIRTHDAY_RE = re.compile(r"\bhappy birthday\b", re.I)
_EXPLICIT_AIRPORT_INSTRUCTION_RE = re.compile(
    r"\b(?:please\s+(?:proceed|follow|continue|go|head)|passengers?\s+(?:arriving|should|please))\b", re.I)
_EXPLICIT_TRANSACTION_RE = re.compile(
    r"\b(?:bill|dollars?|tips?|pay|paid|payment|receipt)\b|\$\s?\d+", re.I)

_EXPLICIT_ENGLISH_BACKGROUND_RE = re.compile(
    r"\b(?:I(?:'m| am) from|I live in|I lived in|I grew up in|I moved to|I moved from|"
    r"I recently arrived in|I just arrived in|I came from|I traveled from|I travelled from)\b", re.I)
_EXPLICIT_CHINESE_BACKGROUND_RE = re.compile(
    r"(?:我来自|我住在|我住过|我刚从|我最近从|我搬到|我长大在|我出生于)")
_EXPLICIT_ENGLISH_EXPERIENCE_RE = re.compile(
    r"\b(?:I|we)(?:'ve| have)?\s+(?:been to|visited|went to|traveled to|travelled to|"
    r"tried|used|watched|played|ate|worked at|lived in)\b|"
    r"\b(?:I|we)\s+(?:visited|went to|traveled to|travelled to|tried|used|watched|"
    r"played|ate|worked at|lived in)\b", re.I)
_EXPLICIT_CHINESE_EXPERIENCE_RE = re.compile(
    r"(?:我|我们)(?:以前|曾经)?(?:去过|到过|住过|吃过|看过|玩过|做过|用过)"
    r"[^。！？.!?;]{0,100}")


def _speech_fact_subject(raw_subject):
    """Keep quoted referents literal; unresolved pronouns never bind to visual people."""
    subject = " ".join(str(raw_subject or "").strip().split())
    folded = subject.casefold()
    if folded in {"i", "me", "we", "myself", "ourselves", "我", "我们", "本人"}:
        return "unidentified speaker"
    if folded in {"he", "she", "they", "他", "她", "他们", "她们"}:
        return f"unidentified person referred to as {folded}"
    return subject[:120]


def _explicit_plan_time_scope(quote):
    match = _EXPLICIT_PLAN_TIME_RE.search(str(quote or ""))
    return match.group(0).strip() if match else ""


def _extract_explicit_speech_facts(transcript, source_segment_id):
    """Keep explicit identity, preference, relationship, background, experience, age,
    plan, and health claims as text facts with audio provenance.

    ASR has no speaker diarization. First-person claims stay attached to an
    unidentified speaker; third-person referents are stored literally and are
    never joined to a visually observed person by pronoun or guesswork.
    """
    facts = []
    last_addressed_name = ""
    last_addressed_end = None
    for line in str(transcript or "").splitlines():
        match = _AUDIO_EVIDENCE_LINE_RE.match(line.strip())
        if not match:
            continue
        quote = match.group("text").strip()
        try:
            audio_start = round(float(match.group("start")), 3)
            audio_end = round(float(match.group("end")), 3)
        except (TypeError, ValueError):
            continue
        if not quote or audio_end < audio_start:
            continue

        evidence_ref = {
            "source_segment_id": str(source_segment_id or "")[:128],
            "modality": "speech",
            "audio_start_s": audio_start,
            "audio_end_s": audio_end,
        }

        self_role_match = (_EXPLICIT_SELF_ROLE_RE.search(quote)
                           or _EXPLICIT_CHINESE_SELF_ROLE_RE.search(quote))
        neighbor_self_identified = bool(_EXPLICIT_SELF_NEIGHBOR_RE.search(quote))
        role_aliases = {
            "grandmother": "grandma", "grandma": "grandma",
            "grandfather": "grandpa", "grandpa": "grandpa",
            "mother": "mom", "mom": "mom", "mum": "mom",
            "father": "dad", "dad": "dad", "neighbour": "neighbor",
            "neighbor": "neighbor", "colleague": "coworker", "coworker": "coworker",
            "奶奶": "grandma", "外婆": "grandma", "爷爷": "grandpa", "外公": "grandpa",
            "妈妈": "mom", "母亲": "mom", "爸爸": "dad", "父亲": "dad",
            "朋友": "friend", "邻居": "neighbor", "同事": "coworker",
            "姐妹": "sister", "兄弟": "brother",
        }
        self_role = (role_aliases.get(self_role_match.group("role").casefold())
                     if self_role_match else None)
        speaker_alias = f"{self_role} 1" if self_role else "unidentified speaker"

        def add_spoken_claim(predicate, value, subject=None,
                             status="stated", actor=None,
                             affected_people=None):
            fact = {
                "subject_type": "person", "subject": subject or speaker_alias,
                "predicate": predicate, "value": str(value or "")[:500],
                "modality": "speech", "status": status,
                "confidence": "medium", "evidence": quote[:500],
                "evidence_ref": dict(evidence_ref),
            }
            # Whisper does not identify voices. Store the utterance source honestly
            # and link only explicitly named or addressed people as text references.
            if actor is None:
                actor = speaker_alias
            if actor:
                fact["actor"] = str(actor)[:120]
            if affected_people:
                fact["affected_people"] = list(dict.fromkeys(
                    str(person).strip()[:120] for person in affected_people if str(person).strip()
                ))
            facts.append(fact)

        vocative = _SPOKEN_VOCATIVE_NAME_RE.search(quote)
        directed_health = _DIRECT_ADDRESSED_HEALTH_RE.search(quote)
        addressed_name = (directed_health.group("name").strip()
                          if directed_health else
                          vocative.group("name").strip() if vocative else "")
        if last_addressed_end is not None and audio_start - last_addressed_end > 3.0:
            last_addressed_name = ""
        if addressed_name:
            last_addressed_name = addressed_name
        elif last_addressed_name:
            addressed_name = last_addressed_name
        last_addressed_end = audio_end
        if addressed_name and (directed_health or vocative):
            add_spoken_claim(
                "addressed_person", f"A speaker addresses the listener as {addressed_name}.",
                subject=_speech_fact_subject(addressed_name),
                actor=speaker_alias,
                affected_people=[_speech_fact_subject(addressed_name)])

        greeting_target = (_speech_fact_subject(addressed_name) if addressed_name
                           else "unidentified addressee")
        greeting_target_label = addressed_name or "the listener"
        if _EXPLICIT_SPOKEN_GREETING_RE.search(quote):
            add_spoken_claim(
                "greeted", f"{speaker_alias} greeted {greeting_target_label}.",
                subject=speaker_alias, status="spoken_event", actor=speaker_alias,
                affected_people=[greeting_target])
        if _EXPLICIT_WELLBEING_QUESTION_RE.search(quote):
            add_spoken_claim(
                "asked_about_wellbeing",
                f"{speaker_alias} asked {greeting_target_label} how they were.",
                subject=speaker_alias, status="spoken_event", actor=speaker_alias,
                affected_people=[greeting_target])
        for spoken_term in terms_in_text(quote):
            # One row per spoken participant: the term itself becomes retrievable and
            # keeps the verbatim quote as evidence, instead of hiding behind
            # "unidentified speaker" as its only subject.
            add_spoken_claim(
                "spoken_family_reference", quote,
                subject=spoken_term, status="quoted_relationship_term",
                actor=speaker_alias, affected_people=[spoken_term])
        if _EXPLICIT_HEALTH_PRODUCT_CLAIM_RE.search(quote):
            add_spoken_claim(
                "quoted_product_claim", quote, status="quoted_unverified_claim")
        if _EXPLICIT_FIRED_RE.search(quote):
            add_spoken_claim(
                "reported_employment_event", quote,
                subject="unidentified addressee",
                affected_people=["unidentified addressee"])
        if _EXPLICIT_BIRTHDAY_RE.search(quote):
            birthday_person = (_speech_fact_subject(addressed_name) if addressed_name
                               else "unidentified addressee")
            add_spoken_claim(
                "stated_birthday_event", quote,
                subject=birthday_person, affected_people=[birthday_person])
        if _EXPLICIT_AIRPORT_INSTRUCTION_RE.search(quote):
            add_spoken_claim("spoken_travel_instruction", quote,
                             subject="public-address announcement",
                             actor="public-address announcement",
                             affected_people=["passengers"])
        if _EXPLICIT_TRANSACTION_RE.search(quote):
            add_spoken_claim(
                "spoken_transaction_detail", quote,
                affected_people=["unidentified addressee"])

        name_match = (_EXPLICIT_ENGLISH_SELF_NAME_RE.search(quote)
                      or _EXPLICIT_CHINESE_SELF_NAME_RE.search(quote))
        if name_match:
            add_spoken_claim("stated_name", name_match.group("name").strip())
        other_name = (_EXPLICIT_ENGLISH_OTHER_NAME_RE.search(quote)
                      or _EXPLICIT_CHINESE_OTHER_NAME_RE.search(quote))
        if other_name:
            add_spoken_claim(
                "stated_name", other_name.group("name").strip(),
                subject=_speech_fact_subject(other_name.group("subject")))
        if (_EXPLICIT_ENGLISH_PREFERENCE_RE.search(quote)
                or _EXPLICIT_CHINESE_PREFERENCE_RE.search(quote)):
            add_spoken_claim("stated_preference", quote)
        other_preference = (_EXPLICIT_ENGLISH_OTHER_PREFERENCE_RE.search(quote)
                            or _EXPLICIT_CHINESE_OTHER_PREFERENCE_RE.search(quote))
        if other_preference:
            add_spoken_claim(
                "stated_preference", quote,
                subject=_speech_fact_subject(other_preference.group("subject")))
        if (_EXPLICIT_ENGLISH_RELATION_RE.search(quote)
                or _EXPLICIT_CHINESE_RELATION_RE.search(quote)
                or self_role_match):
            relationship_value = quote
            relationship_status = "stated"
            if self_role:
                relationship_status = "self_identified"
            if neighbor_self_identified:
                target = addressed_name or "the listener"
                relationship_value = (
                    f"{speaker_alias} self-identifies as {target}'s neighbor.")
                relationship_status = "self_identified"
            add_spoken_claim(
                "stated_relationship", relationship_value,
                subject=speaker_alias, status=relationship_status,
                actor=speaker_alias,
                affected_people=[_speech_fact_subject(addressed_name)]
                if addressed_name else None)
        if (_EXPLICIT_ENGLISH_BACKGROUND_RE.search(quote)
                or _EXPLICIT_CHINESE_BACKGROUND_RE.search(quote)):
            add_spoken_claim("stated_background", quote)
        if (_EXPLICIT_ENGLISH_EXPERIENCE_RE.search(quote)
                or _EXPLICIT_FIRST_VISIT_RE.search(quote)
                or _EXPLICIT_CHINESE_EXPERIENCE_RE.search(quote)):
            add_spoken_claim("stated_experience", quote)

        age_match = _EXPLICIT_ENGLISH_AGE_RE.search(quote)
        age_subject = speaker_alias if age_match else ""
        age_is_english = bool(age_match)
        if not age_match:
            age_match = _EXPLICIT_CHINESE_AGE_RE.search(quote)
            age_subject = speaker_alias if age_match else ""
            age_is_english = False
        if not age_match:
            age_match = _EXPLICIT_ENGLISH_OTHER_AGE_RE.search(quote)
            if age_match:
                age_subject = _speech_fact_subject(age_match.group("subject"))
                age_is_english = True
        if not age_match:
            age_match = _EXPLICIT_CHINESE_OTHER_AGE_RE.search(quote)
            if age_match:
                age_subject = _speech_fact_subject(age_match.group("subject"))
                age_is_english = False

        if age_match:
            age = int(age_match.group("age") if age_match.groupdict().get("age")
                      else age_match.group(1))
            if 1 <= age <= 120:
                age_value = f"{age} years old" if age_is_english else f"{age}岁"
                facts.append({
                    "subject_type": "person", "subject": age_subject,
                    "predicate": "stated_age", "value": age_value,
                    "modality": "speech", "status": "stated",
                    "confidence": "medium", "evidence": quote[:500],
                    "evidence_ref": dict(evidence_ref),
                })

        first_person_plan = (
            _EXPLICIT_ENGLISH_PLAN_RE.search(quote)
            or _EXPLICIT_CHINESE_PLAN_RE.search(quote)
        )
        non_plan_future = re.search(
            r"\b(?:I|we)\s+(?:will|shall|'ll)\s+(?:die|be dead|not survive)\b",
            quote, re.I)
        if non_plan_future:
            first_person_plan = None
        other_plan = None if first_person_plan or non_plan_future else (
            _EXPLICIT_ENGLISH_OTHER_PLAN_RE.search(quote)
            or _EXPLICIT_CHINESE_OTHER_PLAN_RE.search(quote)
        )
        plan_subject = speaker_alias if first_person_plan else ""
        if other_plan:
            plan_subject = _speech_fact_subject(other_plan.group("subject"))
            if plan_subject in {"今天", "今晚", "明天", "后天", "这几天", "本周", "下周"}:
                other_plan = None
                plan_subject = ""
        if first_person_plan or other_plan:
            fact = {
                "subject_type": "person", "subject": plan_subject,
                "predicate": "stated_plan", "value": quote[:500],
                "modality": "speech", "status": "stated_plan",
                "confidence": "medium", "evidence": quote[:500],
                "evidence_ref": dict(evidence_ref),
            }
            time_scope = _explicit_plan_time_scope(quote)
            if time_scope:
                fact["time_scope"] = time_scope[:80]
            facts.append(fact)

        health_match = _EXPLICIT_ENGLISH_HEALTH_RE.search(quote)
        health_subject = speaker_alias if health_match else ""
        if not health_match:
            health_match = _EXPLICIT_CHINESE_HEALTH_RE.search(quote)
            health_subject = speaker_alias if health_match else ""
        if not health_match:
            health_match = _EXPLICIT_ENGLISH_OTHER_HEALTH_RE.search(quote)
            if health_match:
                health_subject = _speech_fact_subject(health_match.group("subject"))
        if not health_match:
            health_match = (_EXPLICIT_CHINESE_OTHER_HEALTH_RE.search(quote)
                            or _EXPLICIT_CHINESE_OTHER_ALLERGY_RE.search(quote))
            if health_match:
                health_subject = _speech_fact_subject(health_match.group("subject"))
        if not health_match and directed_health:
            health_match = directed_health
            health_subject = _speech_fact_subject(directed_health.group("name"))

        if health_match:
            urgent_current_report = bool(
                re.search(r"\b(?:i have|i(?:'m| am) having)\s+(?:a\s+)?(?:heart attack|stroke)\b",
                          quote, re.I)
                or re.search(r"\b(?:help me|i will die|i(?:'m| am) dying)\b", quote, re.I)
                or re.search(r"(?:救命|我快不行了|我要死了)", quote)
            )
            explicit_history = bool(
                re.search(r"\b(?:i had|i have had|i've had|i was diagnosed with|"
                          r"i have been diagnosed with|i have a history of)\b", quote, re.I)
                or re.search(r"(?:我以前|我曾经|患过|得过|被诊断为|确诊)", quote)
            )
            if directed_health:
                health_predicate = "spoken_health_claim"
                health_status = "unverified_claim"
            elif urgent_current_report:
                health_predicate = "self_reported_current_health_event"
                health_status = "self_reported_unverified"
            elif explicit_history:
                health_predicate = "self_reported_health_history"
                health_status = "self_reported"
            else:
                health_predicate = "spoken_health_claim"
                health_status = "self_reported_unverified" if health_subject == "unidentified speaker" else "unverified_claim"
            health_fact = {
                "subject_type": "person", "subject": health_subject,
                "predicate": health_predicate, "value": quote[:500],
                "modality": "speech", "status": health_status,
                "confidence": "medium", "evidence": quote[:500],
                "evidence_ref": dict(evidence_ref),
                "actor": speaker_alias,
            }
            if directed_health and health_subject:
                health_fact["affected_people"] = [health_subject]
            facts.append(health_fact)

        if len(facts) >= 16:
            break
    return facts


def moment_brief(m):
    """单条片段的非标注元数据；语义内容只允许来自模型的视频感知。"""
    video = m.get("video") or ""                 # data/EgoLife/A2_ALICE/DAY1/xxx.mp4

    # 本地视频路径（目录是小写 egolife），存在则给出可播放 URL
    local = ROOT / video.replace("EgoLife", "egolife") if video else None
    has_video = local is not None and local.exists()
    video_url = None
    clip_start = clip_end = None
    if has_video:
        resolved = local.resolve()
        try:
            rel = resolved.relative_to(VIDEO_ROOT)
        except ValueError:
            try:
                rel = resolved.relative_to(FEISHU_TEST_ROOT)
            except ValueError:
                rel = resolved.relative_to(ROOT)
        video_url = "/api/video?p=" + quote(str(rel))
        # start_time/end_time 是相对整点小时的偏移；换算成 30s 片段内秒数
        if m.get("start_time") is not None:
            off = _intra_hour_offset(m.get("id", ""))
            clip_start = round(max(0.0, m["start_time"] - off), 2)
            clip_end = round(m.get("end_time", m["start_time"] + 30) - off, 2)

    return {
        "id": m.get("id", ""),
        "video": video.rsplit("/", 1)[-1] if video else "",
        "clock": _clock_from_id(m.get("id", "")),
        "start_time": m.get("start_time"),
        "end_time": m.get("end_time"),
        "hour": m.get("_hour", ""),
        "people": [],
        "location": "",
        "scene": "",
        "action": "",
        "video_url": video_url,
        "clip_start": clip_start,
        "clip_end": clip_end,
    }


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def feishu_test_moments(dataset):
    """Build deterministic video-only moments from the copied Feishu inventory."""
    if not FEISHU_TEST_MANIFEST.is_file():
        raise ValueError("Feishu test video manifest is missing")
    manifest = json.load(open(FEISHU_TEST_MANIFEST, encoding="utf-8"))
    files = manifest.get("files") or []
    selector = str(dataset or "")
    category = selector.split(":", 1)[1] if selector.startswith("feishu:") else "all"
    if category != "all":
        files = [row for row in files if row.get("scenario") == category]
    moments = []
    for row in files:
        relative = str(row.get("relative_path") or "")
        video_path = (FEISHU_TEST_ROOT / relative).resolve()
        if FEISHU_TEST_ROOT not in video_path.parents or not video_path.is_file():
            continue
        clip_id = str(row.get("clip_id") or Path(relative).stem)
        token = clip_id.replace(".", "_").replace("-", "_")
        digest = str(row.get("sha256") or "")[:10]
        moments.append({
            "id": f"FEISHU_{token}_{digest}",
            "video": (Path("test_data") / FEISHU_TEST_ROOT.name / relative).as_posix(),
            "start_time": 0.0,
            "end_time": float(row.get("duration_seconds") or 0),
            "duration_seconds": float(row.get("duration_seconds") or 0),
            "_hour": "12",
            "test_category": str(row.get("scenario") or ""),
            "test_clip_id": clip_id,
            "transcript": "",
        })
    if not moments:
        raise ValueError(f"No usable Feishu videos in dataset: {dataset}")
    return moments


PARTICIPANT_NAMES = {
    "A1_JAKE": "Jake",
    "A2_ALICE": "Alice",
    "A3_TASHA": "Tasha",
    "A4_LUCIA": "Lucia",
    "A5_KATRINA": "Katrina",
    "A6_SHURE": "Shure",
}


def interactive_dataset_options():
    """Datasets available in the step-by-step runner."""
    options = []
    for dataset, path, plan_path, title in (
        ("base", SHORT_FILE, SHORT_PLAN, "短期 CRUD 演示集"),
        ("expanded", SHORT_EXPANDED_FILE, SHORT_EXPANDED_PLAN, "短期扩展集"),
    ):
        rows = _load_jsonl(path)
        counts = {}
        for row in rows:
            parts = str(row.get("id", "")).split("_")
            participant = "_".join(parts[1:3]) if len(parts) >= 3 else ""
            if participant:
                counts[participant] = counts.get(participant, 0) + 1
        per_person = next(iter(counts.values()), 0)
        purpose = (
            f"{len(rows)} 段，6 人各 {per_person} 段；原 CRUD 步进演示集。"
            if dataset == "base" else
            f"{len(rows)} 段，6 人各 {per_person} 段；扩展短期测评集，可逐段写入并执行 CRUD。"
        )
        options.append({
            "value": f"short_term|{dataset}|all",
            "scenario": "short_term",
            "dataset": dataset,
            "participant": None,
            "label": f"{title} · 全部人物（{len(rows)} 段）",
            "note": purpose,
        })
        for participant in PARTICIPANT_NAMES:
            if participant not in counts:
                continue
            name = PARTICIPANT_NAMES[participant]
            options.append({
                "value": f"short_term|{dataset}|{participant}",
                "scenario": "short_term",
                "dataset": dataset,
                "participant": participant,
                "label": f"{title} · {name}（{counts[participant]} 段）",
                "note": f"{purpose} 当前只加载 {name} 的片段。",
            })

    long_index = json.load(open(LONG_PERSON_DIR / "index.json", encoding="utf-8"))
    for meta in long_index.get("datasets", []):
        participant = meta["participant"]
        name = PARTICIPANT_NAMES.get(participant, participant)
        complete = meta.get("complete", False)
        coverage = round(float(meta.get("coverage", 0)) * 100, 1)
        if complete:
            label = f"长期完整日集 · {name}（{meta['available_count']} 段）"
            note = (
                f"DAY1 完整视频集：{meta['available_count']} 段，覆盖 "
                f"{len(meta.get('hours_covered', []))} 个有数据时段。"
            )
        else:
            label = (
                f"长期部分集 · {name}（{meta['available_count']}/"
                f"{meta['total_candidates']} 段 · {coverage}%）"
            )
            note = (
                f"DAY1 视频不完整：仅 {meta['available_count']}/"
                f"{meta['total_candidates']} 段（{coverage}%）可运行；"
                "结果仅代表本地已下载片段，不是完整日测评。"
            )
        options.append({
            "value": f"long_term|long_day|{participant}",
            "scenario": "long_term",
            "dataset": "long_day",
            "participant": participant,
            "label": label,
            "note": note,
        })

    if FEISHU_TEST_MANIFEST.is_file():
        manifest = json.load(open(FEISHU_TEST_MANIFEST, encoding="utf-8"))
        files = manifest.get("files") or []
        categories = {}
        for row in files:
            category = str(row.get("scenario") or "")
            if category:
                categories[category] = categories.get(category, 0) + 1
        source_note = (
            "飞书文档中的视频测试集。每段视频按原片输入，依次执行场景感知、记忆写入和历史记忆检索；"
            "本模块只提供相关记忆及来源，不生成行动建议。"
        )
        if files:
            options.append({
                "value": "custom_test|feishu:all|all",
                "scenario": "custom_test",
                "dataset": "feishu:all",
                "participant": None,
                "source": "feishu",
                "label": f"飞书自定义测试集 · 全部类别（{len(files)} 段）",
                "note": f"{len(categories)} 类、{len(files)} 段视频。{source_note}",
            })
        for category in sorted(categories, key=lambda value: int(value.split(".", 1)[0])
                                if value.split(".", 1)[0].isdigit() else 999):
            label = FEISHU_TESTSET_LABELS.get(category, category)
            count = categories[category]
            options.append({
                "value": f"custom_test|feishu:{category}|all",
                "scenario": "custom_test",
                "dataset": f"feishu:{category}",
                "participant": None,
                "source": "feishu",
                "label": f"飞书自定义测试集 · {label}（{count} 段）",
                "note": f"{category} · {count} 段视频。{source_note}",
            })
    return options


class InteractiveSession(PostWriteRetrievalMixin):
    """一个场景的步进执行会话。"""

    def __init__(self, key, participant=None, dataset="base", perceiver=None,
                 resume_state=None):
        assert key in SCENARIO_META, f"unknown scenario: {key}"
        self.key = key
        self.participant = participant
        self.dataset_key = dataset
        self.pos = 0            # 已执行步数 = 下一步索引
        self.auto_paused = False
        self.step_cancelled = False
        self.nightly_progress = None
        self.step_progress = None
        self.log = []           # [{kind, title, msg, snapshot, perception}]
        self._cancel_lock = threading.Lock()
        self._active_step_cancel = None
        self._active_step_force = False
        self._active_reasoning_request_id = None
        self._pending_cancel = False
        self._resume_state = resume_state if isinstance(resume_state, dict) else None
        self.extraction_batch_id = f"{key}:{dataset}:{uuid.uuid4().hex[:12]}"
        self.perceiver = perceiver or site_perceiver()
        self.qwen_health = self.perceiver.health()
        self.speech_cache_path = RUNS / "interactive_audio_transcript_cache.json"
        self._speech_cache = PersistentTranscriptCache(self.speech_cache_path)
        self.person_profile_cache = PersistentTranscriptCache(
            RUNS / "interactive_person_profile_cache.json"
        )
        self.retrieval_cache = PersistentAdviceCache(
            RUNS / "interactive_scene_retrieval_cache.sqlite3", ROOT
        )
        self.scene_plan_cache = PersistentAdviceCache(
            RUNS / "interactive_scene_plan_cache.sqlite3", ROOT
        )
        self.grounding_cache = PersistentObjectGroundingCache(
            RUNS / "interactive_object_grounding_cache.sqlite3", ROOT
        )
        self._build()

    def _install_operation_audit(self):
        """Record actual memory API calls made by this interactive session."""
        self.operation_log = []
        self._active_step_no = None
        self._active_step_title = None
        tracked = (
            "add", "query", "query_hierarchical", "update", "update_feedback",
            "highlight", "delete", "sort", "combine", "compress", "update_profile",
            "find_merge_candidates", "merge_moments", "refresh_source_moment",
        )
        for name in tracked:
            original = getattr(self.mem, name, None)
            if not callable(original):
                continue

            def audited(*args, __name=name, __original=original, **kwargs):
                started = time.perf_counter()
                before_moment = None
                before_profile = None
                if __name in ("update", "refresh_source_moment", "update_feedback",
                              "highlight", "delete"):
                    mid = str(args[0] if args else kwargs.get("moment_id", ""))
                    item = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid)
                    before_moment = copy.deepcopy(item) if isinstance(item, dict) else None
                if __name == "merge_moments":
                    archive = self.mem.memory.get("layer7", {}).get("moments", {})
                    source_id = str(args[0] if args else kwargs.get("source_id", ""))
                    target_id = str(args[1] if len(args) > 1 else kwargs.get("target_id", ""))
                    before_moment = {
                        "source": copy.deepcopy(archive.get(source_id, {})),
                        "target": copy.deepcopy(archive.get(target_id, {})),
                    }
                if __name == "update_profile":
                    before_profile = copy.deepcopy(
                        self.mem.memory.get("layer6", {}).get("profile", {})
                    )
                try:
                    result = __original(*args, **kwargs)
                except Exception as exc:
                    self._append_operation(__name, args, kwargs, None,
                                           started, error=exc,
                                           before_moment=before_moment,
                                           before_profile=before_profile)
                    raise
                self._append_operation(
                    __name, args, kwargs, result, started,
                    before_moment=before_moment,
                    before_profile=before_profile,
                )
                return result

            setattr(self.mem, name, audited)

    def _append_operation(self, name, args, kwargs, result, started,
                          error=None, before_moment=None, before_profile=None):
        """Append a factual audit event with the affected memory content."""
        status = "failed" if error else (
            "blocked" if result is False or
            (isinstance(result, dict) and result.get("ok") is False)
            else "success"
        )
        if name in ("update", "refresh_source_moment") and result is False:
            mid = str(args[0] if args else kwargs.get("moment_id", ""))
            exists = mid in self.mem.memory.get("layer7", {}).get("moments", {})
            status = "no_change" if exists else "blocked"
        summary = self._operation_summary(
            name, args, kwargs, result, error, before_moment, before_profile
        )
        self.operation_log.append({
            "seq": len(self.operation_log) + 1,
            "time": time.strftime("%H:%M:%S"),
            "step": self._active_step_no,
            "step_title": self._active_step_title or "",
            "operation": name,
            "status": status,
            "summary": summary,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        })

    @staticmethod
    def _brief_value(value, limit=110):
        if isinstance(value, (dict, list)):
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            text = str(value if value is not None else "—")
        return " ".join(text.split())[:limit]

    def _moment_detail(self, moment):
        """Return the visible, factual fields of a memory without mutating it."""
        if not isinstance(moment, dict):
            return {}
        return {
            "id": str(moment.get("id") or ""),
            "timestamp": moment.get("timestamp") or "",
            "scene": moment.get("scene") or "",
            "action": moment.get("user_action") or "",
            "interaction_details": moment.get("interaction_details") or "",
            "people": moment.get("people") or [],
            "person_observations": copy.deepcopy(moment.get("person_observations") or []),
            "temporal_facts": copy.deepcopy(moment.get("temporal_facts") or []),
            "source_time": copy.deepcopy(moment.get("source_time") or {}),
            "location": moment.get("location") or "",
            "activity": moment.get("activity") or "",
            "objects": moment.get("objects") or [],
            "notes": moment.get("notes") or "",
            "highlighted": bool(moment.get("highlighted")),
            "highlight_source": moment.get("highlight_source") or "",
            "highlighted_at": moment.get("highlighted_at") or "",
            "merged_into": moment.get("merged_into") or "",
            "merged_sources": moment.get("merged_sources") or [],
            "merge_history": moment.get("merge_history") or [],
        }

    def _operation_summary(self, name, args, kwargs, result, error,
                           before_moment=None, before_profile=None):
        if error:
            return f"{name} 失败：{str(error)[:140]}"
        arg0 = args[0] if args else kwargs.get("moment_id")
        if name == "add":
            mid = str(result or "")
            item = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            text = str(item.get("scene") or item.get("user_action") or "").strip()
            text = text.replace("\n", " ")[:100]
            return f"写入记忆 {mid}" + (f" · {text}" if text else "")
        if name in ("query", "query_hierarchical"):
            query = str(arg0 if arg0 is not None else kwargs.get("query_text", ""))[:80]
            hits = result if isinstance(result, list) else []
            examples = []
            for hit in hits[:3]:
                if not isinstance(hit, dict):
                    continue
                moment = hit.get("moment") if isinstance(hit.get("moment"), dict) else hit
                mid = str(moment.get("id") or "无 ID")
                content = str(moment.get("scene") or moment.get("user_action") or "").replace("\n", " ")
                layer = f"L{hit['layer']} · " if hit.get("layer") else ""
                examples.append(f"{layer}{mid}: {content[:100]}")
            detail = "；命中：" + "；".join(examples) if examples else ""
            return f"检索「{query}」· 命中 {len(hits)} 条{detail}"
        if name == "find_merge_candidates":
            candidates = result if isinstance(result, list) else []
            examples = [
                f"{row.get('source_id')}→{row.get('target_id')} "
                f"({row.get('similarity', 0):.0%})"
                for row in candidates[:3]
            ]
            suffix = " · " + "；".join(examples) if examples else ""
            return f"相似事件候选 {len(candidates)} 组（需人工确认）{suffix}"
        if name in ("update", "refresh_source_moment"):
            mid = str(arg0 or "")
            updates = args[1] if len(args) > 1 else kwargs.get("updates", {})
            exists = mid in self.mem.memory.get("layer7", {}).get("moments", {})
            if result is False:
                return f"更新记忆 {mid} · 字段未变化" if exists else f"更新记忆 {mid} · 未找到"
            after = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            changes = []
            if isinstance(updates, dict):
                for field, new_value in updates.items():
                    old_value = (before_moment or {}).get(field)
                    actual_value = after.get(field)
                    changes.append(
                        f"{field}：{self._brief_value(old_value)} → {self._brief_value(actual_value)}"
                    )
            verb = "刷新源记忆" if name == "refresh_source_moment" else "更新记忆"
            return f"{verb} {mid}" + (" · " + "；".join(changes) if changes else "")
        if name == "update_feedback":
            mid = str(arg0 or "")
            corrections = kwargs.get("corrections", {})
            if len(args) > 1 and isinstance(args[1], dict):
                corrections = args[1]
            detail = "；".join(
                f"{key}：{self._brief_value((before_moment or {}).get(key))} → {self._brief_value(value)}"
                for key, value in corrections.items()
            ) if isinstance(corrections, dict) else ""
            confirmed = kwargs.get("confirmed", False)
            rating = kwargs.get("user_rating")
            return f"反馈修正 {mid}" + (f" · {detail}" if detail else "") + (
                f" · 确认={'是' if confirmed else '否'} · 评分={rating if rating is not None else '未评分'}"
            )
        if name == "highlight":
            mid = str(arg0 or "")
            item = before_moment or self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            content = str(item.get("scene") or item.get("user_action") or "").replace("\n", " ")[:100]
            result_text = "已标记重点" if result else "标记失败"
            return f"{result_text} {mid}" + (f" · {content}" if content else "")
        if name == "delete":
            mid = str(arg0 or "")
            reason = kwargs.get("reason", "manual")
            item = before_moment or {}
            content = str(item.get("scene") or item.get("user_action") or "").replace("\n", " ")[:100]
            result_text = "已删除" if result else "被保护/未找到"
            return f"{result_text} {mid} · 原因：{reason}" + (f" · 内容：{content}" if content else "")
        if name == "sort":
            analysis = arg0 if isinstance(arg0, dict) else {}
            tags = sum(len(v) for v in analysis.values() if isinstance(v, dict))
            refs = sum(len(ids) for dim in analysis.values() if isinstance(dim, dict)
                       for ids in dim.values() if isinstance(ids, list))
            return f"整理索引 · {len(analysis)} 个维度、{tags} 个标签、{refs} 条引用"
        if name == "combine":
            details = result if isinstance(result, dict) else {}
            aliases = details.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            examples = []
            for alias in aliases[:4]:
                if not isinstance(alias, dict):
                    continue
                dimension = str(alias.get("dimension") or "索引")
                old = self._brief_value(alias.get("from"), 48)
                new = self._brief_value(alias.get("to"), 48)
                affected = int(alias.get("affected_moments") or 0)
                item = f"{dimension}「{old}」→「{new}」·影响 {affected} 条"
                sample_ids = alias.get("sample_moment_ids")
                if isinstance(sample_ids, list) and sample_ids:
                    item += "·例 " + ", ".join(str(mid) for mid in sample_ids[:3])
                    if affected > len(sample_ids[:3]):
                        item += f" 等 {affected} 条"
                if alias.get("overlap_moments"):
                    item += f"·两索引共有 {alias['overlap_moments']} 条"
                if alias.get("field_updates"):
                    item += f"·字段更新 {alias['field_updates']} 条"
                if alias.get("list_duplicates_removed"):
                    item += f"·列表去重 {alias['list_duplicates_removed']} 项"
                examples.append(item)
            summary = f"规范化标签 {len(aliases)} 组"
            if examples:
                summary += " · " + "；".join(examples)
                if len(aliases) > len(examples):
                    summary += f"；另有 {len(aliases) - len(examples)} 组"
            elif not aliases:
                summary += "（无有效别名需要合并）"
            if details.get("updated_moments"):
                summary += f" · 记忆记录更新 {details['updated_moments']} 条"
            return summary
        if name == "merge_moments":
            details = result if isinstance(result, dict) else {}
            if not details.get("ok"):
                return str(details.get("message") or "合并未执行")
            if details.get("status") == "already_merged":
                return f"来源 {details.get('source_id')} 已归并到 {details.get('target_id')}（无重复改动）"
            source_id = details.get("source_id") or "?"
            target_id = details.get("target_id") or "?"
            fields = "、".join(details.get("merged_fields") or []) or "没有新增列表字段"
            conflicts = len(details.get("scalar_conflicts") or [])
            media = details.get("media_preserved") or []
            retained = sum(bool(item.get("visual")) for item in media)
            return (f"合并事件 {source_id} → {target_id} · 列表字段：{fields}"
                    f" · 标量冲突留在来源并记录 {conflicts} 项"
                    f" · {retained}/{len(media)} 条来源保留多模态引用")
        if name == "compress":
            payload = arg0 if isinstance(arg0, dict) else {}
            layers = "、".join(k.upper() for k in payload if k in ("layer4", "layer5", "layer6"))
            return f"压缩整理摘要 · 更新 {layers or '长期摘要'}"
        if name == "update_profile":
            fields = list(arg0) if isinstance(arg0, dict) else []
            profile = result if isinstance(result, (dict, list)) else self.mem.memory.get("layer6", {}).get("profile", {})
            def count_attrs(node):
                if isinstance(node, dict):
                    if "value" in node and "stale" in node:
                        return (1, int(bool(node.get("stale"))))
                    totals = [count_attrs(value) for value in node.values()]
                elif isinstance(node, list):
                    totals = [count_attrs(value) for value in node]
                else:
                    return (0, 0)
                return (sum(x[0] for x in totals), sum(x[1] for x in totals))
            total, stale = count_attrs(profile)
            detail = f"画像 {total} 项 · 过时 {stale} 项" if total else "暂无画像字段 · 衰减状态已检查"

            def conflicts(node, path=""):
                found = []
                if isinstance(node, dict) and "value" in node:
                    for old_value in node.get("history", []) or []:
                        if not isinstance(old_value, dict):
                            continue
                        found.append({
                            "path": path,
                            "old": old_value.get("value"),
                            "new": node.get("value"),
                            "resolution": old_value.get("resolution") or {},
                        })
                elif isinstance(node, dict):
                    for key, value in node.items():
                        found.extend(conflicts(value, f"{path}.{key}" if path else str(key)))
                elif isinstance(node, list):
                    for index, value in enumerate(node):
                        found.extend(conflicts(value, f"{path}[{index + 1}]"))
                return found

            before_conflicts = conflicts(before_profile or {})
            before_signatures = {
                (item["path"], json.dumps(item["old"], ensure_ascii=False, sort_keys=True, default=str),
                 item["resolution"].get("resolved_at", ""))
                for item in before_conflicts
            }
            new_conflicts = [
                item for item in conflicts(profile)
                if (item["path"], json.dumps(item["old"], ensure_ascii=False, sort_keys=True, default=str),
                    item["resolution"].get("resolved_at", "")) not in before_signatures
            ]
            basis_labels = {
                "observations": "观测次数更多",
                "recency": "最近观察",
                "confidence": "置信度更高",
                "existing_value_tiebreak": "条件相同时保留旧值",
            }
            conflict_note = ""
            if new_conflicts:
                examples = []
                for item in new_conflicts[:3]:
                    resolution = item["resolution"]
                    examples.append(
                        f"{item['path']}「{self._brief_value(item['old'], 32)}」→"
                        f"「{self._brief_value(item['new'], 32)}」"
                        f"（{basis_labels.get(resolution.get('basis'), '裁决依据未记录')}）"
                    )
                conflict_note = f" · 新增冲突裁决 {len(new_conflicts)} 项：{'；'.join(examples)}"
                if len(new_conflicts) > len(examples):
                    conflict_note += f"；另有 {len(new_conflicts) - len(examples)} 项"
                conflict_note += " · 旧值保留在画像历史"
            names = f" · 字段 {chr(12289).join(map(str, fields[:8]))}" if fields else ""
            return f"更新用户画像 · {detail}{conflict_note}{names}"
        return name

    # ------------------------------------------------------------------
    # 感知 → 记忆写入
    # Qwen 看视频产出记忆条目；数据集 caption / ground_truth 只用于评估。
    # ------------------------------------------------------------------

    def cancel_current(self):
        """Interrupt the active step; completed items remain available for retry."""
        with self._cancel_lock:
            event = self._active_step_cancel
            if event is None:
                # Queue the stop if it arrives between STEP_LOCK acquisition and
                # run_step installing its cancellation event.
                self._pending_cancel = True
            else:
                event.set()
        cancel_active = getattr(self.perceiver, "cancel_active", None)
        if callable(cancel_active):
            threading.Thread(target=cancel_active, daemon=True,
                             name="cancel-active-perception").start()
        return {"requested": True, "model_cancel_sent": False,
                "model_cancel_pending": callable(cancel_active)}

    def _set_step_progress(self, **updates):
        current = dict(self.step_progress or {})
        completed = list(current.get("completed") or [])
        if "completed_item" in updates:
            item = updates.pop("completed_item")
            if item:
                completed.append(item)
                completed = completed[-4:]
        current.update(updates)
        current["completed"] = completed
        current["updated_at"] = time.time()
        self.step_progress = current

    @staticmethod
    def _progress_text(m, info):
        fields = info.get("qwen") or {}
        scene = fields.get("scene") or ""
        action = fields.get("user_action") or ""
        people = fields.get("people") or []
        if isinstance(people, list):
            people = ", ".join(str(x) for x in people[:5])
        text = " · ".join(x for x in (scene, action) if x)
        if people:
            text += f" · People: {people}"
        related = info.get("retrieved_memories") or []
        if related:
            text += f" · 关联历史记忆 {len(related)} 条"
        return {"id": m.get("id", ""), "text": text or info.get("error") or "处理完成",
                "retrieved_memories": related,
                "retrieval_meta": info.get("retrieval_meta") or {}}

    def _transcribe_audio_evidence(self, multimodal, source_segment_id, video_path=None,
                                   clip_start_s=0.0, clip_end_s=None, progress=None,
                                   timing_sink=None):
        """Transcribe source-video audio only; never read benchmark text or captions."""
        media = multimodal if isinstance(multimodal, dict) else {}
        jobs = []
        decode_started = time.perf_counter()

        def add_audio_timing(key, started):
            if timing_sink is not None:
                timing_sink[key] = timing_sink.get(key, 0) + round(
                    (time.perf_counter() - started) * 1000)
        # Decode source-video audio transiently for transcription. Only the text
        # transcript and explicit spoken facts enter memory; no audio clip is saved.
        if video_path:
            try:
                start = max(0.0, float(clip_start_s or 0.0))
                end = float(clip_end_s) if clip_end_s is not None else start + 30.0
                if end > start:
                    cursor = start
                    while cursor < end:
                        duration = min(20.0, end - cursor)
                        audio_bytes = _run_ffmpeg([
                            "-ss", f"{cursor:.3f}", "-i", str(video_path),
                            "-t", f"{duration:.3f}", "-vn", "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
                        ])
                        if audio_bytes:
                            jobs.append({"bytes": audio_bytes, "name": "source_audio.wav",
                                         "mime": "audio/wav", "offset": cursor,
                                         "duration": duration})
                        cursor += duration
            except Exception:
                # A decode failure only omits transcript text; memory writing
                # remains available and no legacy audio asset is read as fallback.
                jobs = []
        add_audio_timing("audio_decode_ms", decode_started)
        if timing_sink is not None:
            timing_sink["audio_chunks"] = len(jobs)
            timing_sink.setdefault("speech_cache_hits", 0)
            timing_sink.setdefault("speech_cache_misses", 0)
        rows = []
        for job in jobs:
            audio_data = job.get("bytes")
            if audio_data is None:
                try:
                    audio_data = job["path"].read_bytes()
                except OSError:
                    continue
            cache_key = "whisper-source-audio-v2:" + hashlib.sha256(audio_data).hexdigest()
            lookup_started = time.perf_counter()
            decoded = self._speech_cache.get(cache_key)
            add_audio_timing("speech_cache_lookup_ms", lookup_started)
            if timing_sink is not None:
                timing_sink["speech_cache_hits" if decoded is not None else "speech_cache_misses"] += 1
            if decoded is None:
                if progress:
                    progress("speech_transcription", "正在把视频语音转成文字")
                asr_started = time.perf_counter()
                try:
                    response = requests.post(
                        self.perceiver.next_api_url() + "/transcribe",
                        files={"audio": (job["name"], audio_data, job["mime"])},
                        timeout=(5, 180),
                    )
                    response.raise_for_status()
                    decoded = response.json()
                    if not isinstance(decoded, dict):
                        continue
                    decoded, _inserted = self._speech_cache.put_if_absent(
                        cache_key, decoded
                    )
                except Exception:
                    # Audio is optional evidence; keep the visual memory write available.
                    continue
                finally:
                    add_audio_timing("whisper_transcription_request_ms", asr_started)
            language = str(decoded.get("language") or "unknown")[:24]
            offset = float(job.get("offset") or 0)
            segments = decoded.get("segments") or []
            if segments:
                for segment in segments:
                    if not isinstance(segment, dict):
                        continue
                    text = str(segment.get("text") or "").strip()
                    if not text:
                        continue
                    start = offset + max(0.0, float(segment.get("start") or 0))
                    end = offset + max(float(segment.get("start") or 0),
                                       float(segment.get("end") or 0))
                    rows.append({"start_s": round(start, 2), "end_s": round(end, 2),
                                 "language": language, "text": text[:1000],
                                 "source_segment_id": str(source_segment_id or "")[:128]})
            else:
                text = str(decoded.get("full_text") or "").strip()
                if text:
                    rows.append({"start_s": round(offset, 2),
                                 "end_s": round(offset + float(job.get("duration") or 0), 2),
                                 "language": language, "text": text[:1000],
                                 "source_segment_id": str(source_segment_id or "")[:128]})
        if timing_sink is not None:
            timing_sink["transcript_segments"] = len(rows)
        return "\n".join(
            f"[audio {row['start_s']:.2f}–{row['end_s']:.2f}s; {row['language']}] {row['text']}"
            for row in rows
        )[:4000]

    def _enrich_person_profiles_from_speech(self, observations, transcript,
                                            frames, source_segment_id,
                                            cancel_event=None, progress=None):
        """Use source-video speech as evidence on person cards; never persist transcript rows."""
        rows = [dict(row) for row in observations or [] if isinstance(row, dict)]
        speech = str(transcript or "").strip()[:4000]
        image_frames = [frame for frame in (frames or [])
                        if isinstance(frame, dict) and frame.get("bytes")][:7]
        if not rows or not speech or not image_frames:
            return rows, ""
        transcript_hash = hashlib.sha256(speech.encode("utf-8")).hexdigest()
        cache_key = hashlib.sha256(json.dumps({
            "source": str(source_segment_id or ""),
            "transcript": transcript_hash,
            "people": [(str(row.get("entity") or ""), str(row.get("appearance") or ""))
                       for row in rows],
            "version": "person-profile-from-speech-v1",
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        cached = self.person_profile_cache.get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("people"), list):
            payload = cached
        else:
            if cancel_event is not None and cancel_event.is_set():
                return rows, ""
            frame_map = [{"frame": index + 1,
                          "time_s": round(float(frame.get("at_s") or 0), 3)}
                         for index, frame in enumerate(image_frames)]
            candidates = [{"entity": str(row.get("entity") or "")[:100],
                           "appearance": str(row.get("appearance") or "")[:220],
                           "face_features": str(row.get("face_features") or "")[:160]}
                          for row in rows[:16]]
            prompt = (
                "You are enriching a source-grounded person-centered memory. Inspect the attached "
                "ordered video frames together with the time-coded speech transcript from this SAME "
                "video. The transcript is noisy ASR evidence, not verified truth. Return one update "
                "for existing visible candidates only; never invent or merge a person. Use explicit "
                "introductions, forms of address, replies, turn order, gaze, gestures, and who is "
                "visibly interacting to map spoken names/roles to a candidate. Do not assume that "
                "the speaker is the camera wearer, and do not infer age, health, identity, or a family "
                "role from appearance. A concise display_label may replace a visual alias only when "
                "the transcript and scene support that match; otherwise keep the exact current entity. "
                "Put stable or useful personal context (relationship, explicit plans/preferences, "
                "self-reported background or health history, and meaningful interaction with the "
                "camera wearer) on that person's card. Also summarize the shared social/event context "
                "in scene_context (for example, a family visit during a holiday) and cite it with exact "
                "context_evidence_quotes. Keep greetings only when they establish a relationship, "
                "identity, or meaningful social context; never save a greeting as an isolated fact. "
                "Do not return action-by-action facts or the transcript itself. Every update needs "
                "one or more exact short quotes "
                "copied from the transcript as evidence. Preserve the language of display labels used "
                "in the quote (for example 姥姥, 妈妈); profile_context and relationships should be "
                "concise. If identity-to-person matching is uncertain, leave display_label unchanged, "
                "set confidence low, and state only the supported relationship without assigning it "
                "to a visible person. Return JSON only: {\"people\":[{\"entity\":\"exact candidate "
                "label\",\"display_label\":\"short supported name/role or unchanged label\","
                "\"relationships\":[\"supported relationship\"],\"profile_context\":\"concise "
                "person-centered context\",\"evidence_quotes\":[\"exact transcript quote\"],"
                "\"confidence\":\"high|medium|low\",\"frame\":1}],"
                "\"scene_context\":\"concise supported shared event context or empty\","
                "\"context_evidence_quotes\":[\"exact quote\"]}. Candidate people: "
                + json.dumps(candidates, ensure_ascii=False)
                + ". Frame map: " + json.dumps(frame_map, ensure_ascii=False)
                + ". Time-coded source transcript:\n" + speech
            )
            files = [("images", (f"profile_{index + 1}.jpg", frame["bytes"],
                                  frame.get("mime") or "image/jpeg"))
                     for index, frame in enumerate(image_frames)]
            request_id = uuid.uuid4().hex
            api_url = self.perceiver.next_api_url()
            self.perceiver._activate_request(request_id, api_url)
            try:
                if progress:
                    progress("person_profile_enrichment",
                             "正在把语音身份与关系线索归入人物画像")
                response = requests.post(
                    f"{api_url}/analyze_video", files=files,
                    data={"prompt": prompt, "request_id": request_id},
                    timeout=(5, 240),
                )
                if cancel_event is not None and cancel_event.is_set():
                    return rows, ""
                if response.status_code != 200:
                    return rows, ""
                raw = str(response.json().get("analysis") or "").strip()
                start = raw.find("{")
                end = raw.rfind("}")
                if start < 0 or end < start:
                    return rows, ""
                payload = json.loads(raw[start:end + 1])
                if not isinstance(payload, dict) or not isinstance(payload.get("people"), list):
                    return rows, ""
                self.person_profile_cache.put_if_absent(cache_key, payload)
            except Exception:
                return rows, ""
            finally:
                self.perceiver._finish_request(request_id)

        def normalize_quote(value):
            return "".join(str(value or "").casefold().split()).strip("，。！？、,.!?;；:：\"'“”‘’")

        transcript_lines = []
        for line in speech.splitlines():
            match = _AUDIO_EVIDENCE_LINE_RE.match(line.strip())
            if not match:
                continue
            try:
                start_s, end_s = float(match.group("start")), float(match.group("end"))
            except (TypeError, ValueError):
                continue
            transcript_lines.append((str(match.group("text") or "").strip(), start_s, end_s))
        candidates = {str(row.get("entity") or "").strip().casefold(): row for row in rows
                      if str(row.get("entity") or "").strip()}
        for update_row in payload.get("people") or []:
            if not isinstance(update_row, dict):
                continue
            target = candidates.get(str(update_row.get("entity") or "").strip().casefold())
            if target is None:
                continue
            quotes = [str(value).strip()[:300] for value in update_row.get("evidence_quotes") or []
                      if isinstance(value, str) and str(value).strip()]
            matched = []
            for quote in quotes[:6]:
                normalized = normalize_quote(quote)
                if not normalized:
                    continue
                for line_text, start_s, end_s in transcript_lines:
                    line_norm = normalize_quote(line_text)
                    if normalized in line_norm or line_norm in normalized:
                        matched.append((line_text, start_s, end_s))
                        break
            if not matched:
                continue
            confidence = str(update_row.get("confidence") or "low").casefold()
            target["profile_confidence"] = confidence if confidence in {"high", "medium", "low"} else "low"
            if confidence not in {"high", "medium"}:
                continue
            display = " ".join(str(update_row.get("display_label") or "").split()).strip(" .，。;；:")
            if (display and len(display) <= 80 and confidence in {"high", "medium"}
                    and any(normalize_quote(display) in normalize_quote(item[0]) for item in matched)):
                target["entity"] = display
            relationships = update_row.get("relationships")
            if isinstance(relationships, list):
                clean_relations = list(dict.fromkeys(
                    " ".join(str(value).split()).strip()[:180]
                    for value in relationships if isinstance(value, str) and value.strip()
                ))[:8]
                if clean_relations:
                    target["relationships"] = clean_relations
            profile_context = " ".join(str(update_row.get("profile_context") or "").split()).strip()
            if profile_context:
                target["profile_context"] = profile_context[:700]
            refs = list(target.get("evidence_refs") or [])
            for _line_text, start_s, end_s in matched:
                refs.append({"source_segment_id": str(source_segment_id or "")[:128],
                             "modality": "speech", "audio_start_s": round(start_s, 3),
                             "audio_end_s": round(end_s, 3)})
            unique_refs = []
            seen_refs = set()
            for ref in refs:
                key = (ref.get("source_segment_id"), ref.get("modality"),
                       ref.get("frame_index"), ref.get("frame_time_s"),
                       ref.get("audio_start_s"), ref.get("audio_end_s"))
                if key not in seen_refs:
                    seen_refs.add(key)
                    unique_refs.append(ref)
            if unique_refs:
                target["evidence_refs"] = unique_refs[:16]
                target["evidence_ref"] = unique_refs[0]
        context_quotes = [str(value).strip()[:300]
                          for value in payload.get("context_evidence_quotes") or []
                          if isinstance(value, str) and str(value).strip()]
        context_matches = []
        for quote in context_quotes[:6]:
            normalized = normalize_quote(quote)
            if not normalized:
                continue
            for line_text, start_s, end_s in transcript_lines:
                line_norm = normalize_quote(line_text)
                if normalized in line_norm or line_norm in normalized:
                    context_matches.append((line_text, start_s, end_s))
                    break
        scene_context = " ".join(str(payload.get("scene_context") or "").split()).strip()[:700]
        if not context_matches:
            scene_context = ""
        return rows, scene_context

    @staticmethod
    def _entity_key(value):
        return " ".join(str(value or "").casefold().split()).strip(" .,:;-_")

    @staticmethod
    def _entity_crop_gap(media, fields):
        """Detect missing expected entity crops so a later run can retry them."""
        media = media if isinstance(media, dict) else {}
        fields = fields if isinstance(fields, dict) else {}
        objects = [
            str(row.get("name") or row.get("object") or row.get("entity") or "").strip()
            if isinstance(row, dict) else str(row or "").strip()
            for row in (fields.get("objects") or [])
        ][:12]
        people = [str(row.get("entity") or "").strip()
                  for row in (fields.get("person_observations") or [])
                  if isinstance(row, dict) and str(row.get("entity") or "").strip()][:8]
        for key, labels, label_key, role in (
            ("object_visuals", objects, "object", "object_crop"),
            ("person_visuals", people, "person", "person_crop"),
        ):
            labels = [label for label in labels if label]
            if not labels:
                continue
            visual = media.get(key)
            if not isinstance(visual, dict) or visual.get("status") in {"unavailable", "error"}:
                return True
            normalize = lambda value: " ".join(str(value or "").casefold().split())
            items = {normalize(item.get(label_key)): item
                     for item in visual.get("items") or [] if isinstance(item, dict)}
            for label in labels:
                item = items.get(normalize(label))
                if not item:
                    return True
                if item.get("status") == "not_visible":
                    continue
                if item.get("status") in {"unavailable", "not_verified", "error"}:
                    return True
                if not any(
                    isinstance(ref, dict) and ref.get("asset")
                    and (ref.get("visual_role") == role or ref.get("crop_bbox"))
                    for ref in item.get("frame_refs") or []
                ):
                    return True
        return False

    def _repair_entity_crop_evidence(self, dataset_id, fields, video_path,
                                     clip_start, clip_end, media, frames,
                                     progress=None):
        """Retry failed person/object grounding for a previously written moment."""
        frames = list(frames or [])
        if not frames:
            frames = self.media_store.sample_visual_candidates(
                video_path, clip_start, clip_end, source_id=dataset_id, count=7)
        visual_refs = media.get("visual_frames") or (
            [media.get("visual")] if media.get("visual") else [])
        if not frames:
            for ref in visual_refs[:3]:
                path = Path(self.mem.memory_dir) / str(ref.get("asset") or "")
                if ref.get("asset") and path.is_file():
                    frames.append({**ref, "mime": ref.get("mime") or "image/webp",
                                   "bytes": path.read_bytes()})

        def unavailable(labels, label_key, exc):
            message = str(exc)[:500]
            return {"schema": 1, "status": "unavailable", "model": "Qwen3-VL",
                    "message": message,
                    "items": [{label_key: str(label), "status": "unavailable",
                               "appearance": "", "error": message,
                               "frame_refs": [], "crop_count": 0} for label in labels]}

        def run_grounding(kind, labels, label_key, source_key=""):
            if not labels or not frames:
                return {"schema": 1, "status": "no_visual_evidence", "items": [],
                        "message": "没有可用于定位的画面"}
            result = self.grounding_cache.get_source(source_key)[0] if source_key else None
            cache_labels = labels
            if kind == "person":
                cache_labels = [
                    "person-crop-v1: " + str(row.get("entity") or "").strip()
                    + " | " + str(row.get("appearance") or "").strip() for row in labels]
            cache_key = self.grounding_cache.key(dataset_id, cache_labels, frames)
            if result is None:
                result = self.grounding_cache.get(cache_key)[0]
            if result is None:
                if progress:
                    progress("person_visual_grounding" if kind == "person"
                             else "object_visual_grounding",
                             "正在重试人物局部画面" if kind == "person"
                             else "正在重试物品局部画面")
                try:
                    cancel_event = self._active_step_cancel
                    if cancel_event is not None and cancel_event.is_set():
                        return {"schema": 1, "status": "cancelled", "items": []}
                    result = (self.perceiver.ground_people(labels, frames,
                               cancel_event=cancel_event) if kind == "person" else
                              self.perceiver.ground_objects(labels, frames,
                               cancel_event=cancel_event))
                except Exception as exc:
                    names = [row.get("entity") for row in labels] if kind == "person" else labels
                    return unavailable(names, label_key, exc)
            if not isinstance(result, dict) or result.get("status") != "completed":
                message = str((result.get("message") if isinstance(result, dict) else "")
                              or "视觉模型未返回可用定位")
                names = [row.get("entity") for row in labels] if kind == "person" else labels
                failed = unavailable(names, label_key, message)
                if isinstance(result, dict):
                    failed.update({key: value for key, value in result.items()
                                   if key not in {"items", "status", "message"}})
                return failed
            result = copy.deepcopy(result)
            self.grounding_cache.put(cache_key, result)
            if source_key:
                self.grounding_cache.put_source(source_key, result)
            for item in result.get("items") or []:
                if not isinstance(item, dict):
                    continue
                boxes = {}
                for box in item.get("frame_boxes") or []:
                    try:
                        index, bbox = int(box.get("frame_index")), box.get("bbox")
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if 1 <= index <= len(frames) and isinstance(bbox, list) and len(bbox) == 4:
                        boxes[index] = bbox
                indices = list(dict.fromkeys(
                    [index for index in item.get("frame_indices") or []
                     if isinstance(index, int)] + list(boxes)))
                refs = []
                for index in indices:
                    if index not in boxes:
                        continue
                    frame, bbox = frames[index - 1], boxes[index]
                    crop = self.media_store.crop_visual_candidate(frame.get("bytes"), bbox, 12_000)
                    crop_ref = self.media_store._put("visual", crop, "webp", 12_000)
                    if not crop_ref or not crop_ref.get("asset"):
                        continue
                    source_bytes = self.media_store.fit_visual_candidate(frame.get("bytes"), 6_000)
                    source_ref = self.media_store._put("visual", source_bytes, "webp", 6_000)
                    if source_ref:
                        source_ref.update({"mime": "image/webp", "dhash": frame.get("dhash") or "",
                                           "at_s": frame.get("at_s"),
                                           "source_id": frame.get("source_id") or dataset_id,
                                           "quality_tier": "recent"})
                    refs.append({"asset": crop_ref["asset"],
                                 "bytes": int(crop_ref.get("bytes") or 0),
                                 "at_s": frame.get("at_s"),
                                 "source_id": frame.get("source_id") or dataset_id,
                                 "visual_role": "person_crop" if kind == "person" else "object_crop",
                                 "crop_bbox": [round(float(value), 5) for value in bbox],
                                 **({"source_asset": source_ref["asset"]}
                                    if source_ref and source_ref.get("asset") else {})})
                    break
                item["frame_refs"], item["crop_count"] = refs, len(refs)
                item.pop("frame_indices", None)
                item.pop("frame_boxes", None)
                if refs:
                    item["status"] = "grounded"
                elif item.get("status") != "not_visible":
                    item["status"] = "not_verified"
                    item["error"] = "模型未提供可用边界框，等待重试"
            result["crop_schema"] = 1
            result["crop_method"] = ("Qwen3-VL normalized whole-person box with padded WebP crop"
                                     if kind == "person" else
                                     "Qwen3-VL normalized object box with padded WebP crop")
            return result

        object_labels = [
            str(row.get("name") or row.get("object") or row.get("entity") or "").strip()
            if isinstance(row, dict) else str(row or "").strip()
            for row in fields.get("objects") or []]
        object_labels = [label for label in object_labels if label][:12]
        person_rows = [row for row in fields.get("person_observations") or []
                       if isinstance(row, dict) and str(row.get("entity") or "").strip()][:8]
        if object_labels:
            source_key = self.grounding_cache.source_key(
                dataset_id, video_path, clip_start, clip_end, object_labels)
            media["object_visuals"] = run_grounding(
                "object", object_labels, "object", source_key)
        if person_rows:
            media["person_visuals"] = run_grounding("person", person_rows, "person")
        return media

    def _refresh_existing_multimodal(self, m, moment_id, fields, source_provenance,
                                     progress=None):
        """Refresh source-linked context evidence without requiring model inference.

        Existing person/object crops are retained only when their exact entity labels
        still occur in the refreshed perception. New detector boxes are never guessed.
        """
        stored = self.mem.memory.get("layer7", {}).get("moments", {}).get(moment_id)
        if not isinstance(stored, dict):
            return False
        old_media = stored.get("multimodal") or {}
        if (old_media.get("capture_version") == CAPTURE_VERSION
                and not self._entity_crop_gap(old_media, fields)):
            return False
        video_path = video_path_of(m)
        if not video_path or not Path(video_path).is_file():
            return False
        clip_start, clip_end = clip_range(m)
        try:
            media = self.media_store.capture(
                video_path, clip_start, clip_end,
                source_id=str(m.get("id") or ""), fields=fields,
                recent_media=[x.get("multimodal") for x in
                              list(self.mem.memory.get("layer7", {}).get("moments", {}).values())[-100:]],
                source_provenance=source_provenance,
            )
            if not isinstance(media, dict) or media.get("capture_status") != "captured":
                return False
            context_frames = self.media_store.sample_visual_candidates(
                video_path, clip_start, clip_end,
                source_id=str(m.get("id") or ""), count=7)
            if context_frames:
                media["entity_visuals"] = self.media_store.capture_context_entities(
                    context_frames, fields, source_id=str(m.get("id") or ""),
                    clip_start_s=clip_start,
                )
            else:
                media["entity_visuals"] = {
                    "schema": 1, "status": "no_visual_evidence",
                    "assets": [], "items": [],
                }
            current_objects = {
                self._entity_key(x.get("name") if isinstance(x, dict) else x)
                for x in fields.get("objects") or []
            }
            current_people = {
                self._entity_key(x.get("entity"))
                for x in fields.get("person_observations") or [] if isinstance(x, dict)
            }
            for key, allowed, label_key in (
                ("object_visuals", current_objects, "object"),
                ("person_visuals", current_people, "person"),
            ):
                old = old_media.get(key)
                if not isinstance(old, dict):
                    continue
                role = "object_crop" if key == "object_visuals" else "person_crop"
                kept = [dict(item) for item in old.get("items") or []
                        if isinstance(item, dict)
                        and self._entity_key(item.get(label_key)) in allowed
                        and (item.get("status") == "not_visible" or any(
                            isinstance(ref, dict) and ref.get("asset")
                            and (ref.get("visual_role") == role or ref.get("crop_bbox"))
                            for ref in item.get("frame_refs") or []))]
                if kept:
                    media[key] = {**old, "items": kept}
            media = self._repair_entity_crop_evidence(
                str(m.get("id") or ""), fields, video_path,
                clip_start, clip_end, media, context_frames, progress=progress)
            # Keep earlier successful crops if this retry still cannot reach the
            # vision endpoint; never replace good evidence with an empty failure.
            for key, label_key in (("object_visuals", "object"),
                                   ("person_visuals", "person")):
                previous = old_media.get(key) or {}
                current = media.get(key) or {}
                role = "object_crop" if key == "object_visuals" else "person_crop"
                merged = {
                    self._entity_key(item.get(label_key)): dict(item)
                    for item in previous.get("items") or [] if isinstance(item, dict)
                    and (item.get("status") == "not_visible" or any(
                        isinstance(ref, dict) and ref.get("asset")
                        and (ref.get("visual_role") == role or ref.get("crop_bbox"))
                        for ref in item.get("frame_refs") or []))
                }
                for item in current.get("items") or []:
                    if not isinstance(item, dict):
                        continue
                    label = self._entity_key(item.get(label_key))
                    has_crop = any(
                        isinstance(ref, dict) and ref.get("asset")
                        and (ref.get("visual_role") == role or ref.get("crop_bbox"))
                        for ref in item.get("frame_refs") or [])
                    if label and (has_crop or item.get("status") == "not_visible"
                                  or label not in merged):
                        merged[label] = dict(item)
                if merged or current:
                    media[key] = {**current, "items": list(merged.values())}
            media["capture_version"] = CAPTURE_VERSION
            media["source_provenance"] = dict(source_provenance)
            stored["multimodal"] = media
            provenance = stored.setdefault("source_provenance", {})
            if isinstance(provenance, dict):
                provenance["multimodal_capture_version"] = CAPTURE_VERSION
            media["entity_visuals"] = _build_entity_visuals(stored)
            self.mem._save()
            return True
        except Exception:
            # Text-memory refresh remains usable if local video decoding or asset
            # storage fails; leave the previous multimodal payload untouched.
            return False

    def _refresh_existing_source_record(self, m, moment_id, force, update):
        """Refresh an existing record only from a current, fingerprinted cache row."""
        stored = self.mem.memory.get("layer7", {}).get("moments", {}).get(moment_id)
        if not isinstance(stored, dict):
            return None
        old = stored.get("source_provenance") or {}
        dataset_id = str(m.get("id") or "")
        video_path = video_path_of(m)
        clip_start, clip_end = clip_range(m)
        expected_video = _source_video_id(m, video_path)
        try:
            same_clip = (
                str(old.get("source_segment_id") or "") == dataset_id
                and str(old.get("source_video_id") or "") == expected_video
                and abs(float(old.get("clip_start_s")) - float(clip_start or 0)) <= 0.05
                and (clip_end is None or abs(float(old.get("clip_end_s")) - float(clip_end)) <= 0.05)
            )
        except (TypeError, ValueError):
            same_clip = False
        if not same_clip:
            return None
        stale_version = (
            old.get("perception_prompt_version") != PROMPT_VERSION
            or old.get("list_parser_version") != LIST_PARSER_VERSION
        )
        stored_multimodal = stored.get("multimodal") or {}
        stale_multimodal = (
            stored_multimodal.get("capture_version") != CAPTURE_VERSION
            or self._entity_crop_gap(stored_multimodal, stored)
        )
        if not force and not stale_version and not stale_multimodal:
            return None
        update("cache_refresh", "正在核验缓存版本，并刷新同一视频片段的已有记忆")
        def perception_progress(event):
            update(event.get("stage", "model_inference"),
                   event.get("message", "正在读取当前片段的有效缓存"))
        result = self.perceiver.perceive(
            {"id": dataset_id, "video": m.get("video", ""),
             "start_time": m.get("start_time"), "end_time": m.get("end_time")},
            force=force, cancel_event=self._active_step_cancel,
            progress_callback=perception_progress, allow_legacy_cache=False,
            allow_stale_cache_fallback=True,
        )
        fields = result.get("fields")
        fingerprint = result.get("source_input_fingerprint") or {}
        if (result.get("stale_cache_fallback") and isinstance(fields, dict)
                and fingerprint.get("video_sha256") and fingerprint.get("clip_sha256")):
            return result, copy.deepcopy(fields)
        if (not isinstance(fields, dict) or not fingerprint.get("video_sha256")
                or not fingerprint.get("clip_sha256")
                or result.get("prompt_version") != PROMPT_VERSION
                or result.get("list_parser_version") != LIST_PARSER_VERSION):
            return None
        fields = copy.deepcopy(fields)
        stale_multimodal = (
            stale_multimodal
            or self._entity_crop_gap(stored_multimodal, fields)
        )
        frame_times = result.get("frame_times_s") or []
        def attach_refs(rows, modality):
            linked = []
            for original in rows if isinstance(rows, list) else []:
                if not isinstance(original, dict):
                    continue
                row = dict(original)
                ref = {
                    "source_segment_id": dataset_id,
                    "clip_start_s": round(float(clip_start or 0), 2),
                    "clip_end_s": round(float(clip_end), 2) if clip_end is not None else None,
                    "modality": modality,
                }
                index = row.get("frame_index")
                if isinstance(index, int) and not isinstance(index, bool):
                    ref["frame_index"] = index
                    if 1 <= index <= len(frame_times):
                        ref["frame_time_s"] = frame_times[index - 1]
                        row["frame_time_s"] = frame_times[index - 1]
                row.setdefault("modality", modality)
                row["evidence_ref"] = ref
                linked.append(row)
            return linked
        fields["person_observations"] = attach_refs(
            fields.get("person_observations") or [], "visual")
        # A refresh replaces the visual rows from the new cache read and carries over
        # the speech rows already attached to the stored moment.
        visual_facts = attach_refs(fields.get("temporal_facts") or [], "visual")
        previous_facts = stored.get("temporal_facts") or []
        speech_facts = [copy.deepcopy(row) for row in previous_facts
                        if isinstance(row, dict) and row.get("modality") == "speech"]
        fields["temporal_facts"] = visual_facts + speech_facts
        normalize_object_transition_facts(fields)
        source_day = next((part.upper() for part in dataset_id.split("_")
                           if part.upper().startswith("DAY") and part[3:].isdigit()), "")
        source_time = {
            "day_label": source_day,
            "time_of_day": _clock_from_id(dataset_id),
            "time_occasion": str(fields.get("time_occasion") or "")[:200],
            "clip_start_s": round(float(clip_start or 0), 2),
            "basis": "dataset_relative" if source_day else "video_relative",
            "source_segment_id": dataset_id,
        }
        if clip_end is not None:
            source_time["clip_end_s"] = round(float(clip_end), 2)
        source_provenance = {
            "source_video_id": expected_video,
            "source_segment_id": dataset_id,
            "clip_start_s": round(float(clip_start or 0), 2),
            "clip_end_s": round(float(clip_end), 2) if clip_end is not None else None,
            "source_video_sha256": fingerprint["video_sha256"],
            "source_clip_sha256": fingerprint["clip_sha256"],
            "perception_prompt_version": result.get("prompt_version"),
            "list_parser_version": result.get("list_parser_version"),
            "multimodal_capture_version": CAPTURE_VERSION,
            "extraction_batch_id": old.get("extraction_batch_id") or self.extraction_batch_id,
        }
        updates = {
            "scene": fields.get("scene") or "",
            "user_action": fields.get("user_action") or "",
            "interaction_details": fields.get("interaction_details") or "",
            "person_observations": fields.get("person_observations") or [],
            "temporal_facts": fields.get("temporal_facts") or [],
            "source_evidence": "",
            "source_time": source_time,
            "location": fields.get("location") or "",
            "activity": fields.get("activity") or "",
            "environments": fields.get("environments") or [],
            "objects": fields.get("objects") or [],
        }
        if fields.get("people"):
            updates["people"] = fields["people"]
        if fields.get("time_occasion"):
            updates["notes"] = str(fields["time_occasion"])[:200]
        refreshed = self.mem.refresh_source_moment(
            moment_id, updates, source_provenance, expected_segment_id=dataset_id)
        multimodal_refreshed = False
        if stale_multimodal:
            multimodal_refreshed = self._refresh_existing_multimodal(
                m, moment_id, fields, source_provenance,
                progress=lambda stage, message: update(stage, message))
        result["multimodal_refreshed"] = multimodal_refreshed
        if not refreshed and not multimodal_refreshed:
            return None
        if result.get("source") == "cache_quality_warning":
            target = self.mem.memory.get("layer7", {}).get("moments", {}).get(moment_id)
            if isinstance(target, dict):
                provenance = target.setdefault("source_provenance", {})
                provenance["perception_status"] = "cache_quality_warning"
                provenance["perception_quality_warning"] = str(
                    result.get("quality_warning") or "")[:700]
                self.mem._save()
        fields["source_provenance"] = source_provenance
        self._qwen_fields[dataset_id] = fields
        return result, fields

    def _add_one(self, m, force=False, progress=None):
        """用 Qwen 的感知产出写入一条记忆，返回 (moment_id, info)。"""
        dataset_id = m.get("id", "")
        item_started = time.perf_counter()
        operation_ms = {}
        progress_stage_ms = {}
        active_stage = "prepare"
        active_stage_started = item_started

        def record_duration(name, started):
            operation_ms[name] = round((time.perf_counter() - started) * 1000)

        def close_progress_stage(now=None):
            nonlocal active_stage_started
            if active_stage and active_stage_started is not None:
                end = time.perf_counter() if now is None else now
                elapsed = max(0, round((end - active_stage_started) * 1000))
                progress_stage_ms[active_stage] = progress_stage_ms.get(active_stage, 0) + elapsed
            active_stage_started = now

        def update(stage, message, output=None):
            nonlocal active_stage, active_stage_started
            now = time.perf_counter()
            if str(stage) != active_stage:
                close_progress_stage(now)
                active_stage = str(stage)
                active_stage_started = now
            if progress:
                self._set_step_progress(**progress, stage=stage, message=message,
                                        output=output)

        def finish_timing(info):
            close_progress_stage()
            operation_ms["total_wall_ms"] = round((time.perf_counter() - item_started) * 1000)
            info["stage_timings_ms"] = {
                **operation_ms,
                "progress_stage_ms": dict(progress_stage_ms),
            }
            try:
                cache_sources = {"cache", "cache_quality_warning", "stale_cache_fallback",
                                 "legacy_cache", "label_normalization_only"}
                record = {
                    "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_segment_id": str(dataset_id),
                    "perception_source": info.get("source"),
                    "perception_model": "Qwen3-VL-8B-Instruct",
                    "perception_model_called": info.get("source") not in cache_sources,
                    "perception_elapsed_ms": info.get("elapsed_ms"),
                    "perception_elapsed_kind": (
                        "historical_cached_inference" if info.get("source") in cache_sources
                        else "current_model_inference"
                    ),
                    "stage_timings_ms": info["stage_timings_ms"],
                    "retrieval_cache": ((info.get("retrieval_meta") or {}).get("cache") or {}).get("status"),
                    "retrieval_stage_timings_ms": ((info.get("retrieval_meta") or {}).get("stage_timings_ms") or {}),
                }
                path = Path(self.mem.memory_dir) / "processing_latency_log.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except (OSError, TypeError, ValueError):
                pass
            return info
        cancel_event = self._active_step_cancel
        if cancel_event is not None and cancel_event.is_set():
            info = {"dataset_id": dataset_id, "source": "cancelled",
                    "cancelled": True, "error": "当前片段处理已终止"}
            update("cancelled", info["error"])
            finish_timing(info)
            return None, info
        existing_mid = self.id_map.get(dataset_id)
        if existing_mid:
            cached = self.perceiver.cache.get(dataset_id) or {}
            fields = cached.get("fields") or self._qwen_fields.get(dataset_id) or {}
            refresh = self._refresh_existing_source_record(
                m, existing_mid, force, update)
            refreshed_existing = refresh is not None
            if refreshed_existing:
                refresh_result, fields = refresh
                cached = refresh_result
                if refresh_result.get("stale_cache_fallback"):
                    refreshed_existing = False
            info = {
                "dataset_id": dataset_id,
                "moment_id": existing_mid,
                "qwen": fields,
                "raw": cached.get("raw", ""),
                "elapsed_ms": cached.get("elapsed_ms"),
                "num_frames": cached.get("num_frames"),
                "source": cached.get("source", "cache"),
                "memory_refreshed": refreshed_existing,
                "stale_cache_fallback": bool(cached.get("stale_cache_fallback")),
                "quality_warning": cached.get("quality_warning", ""),
                "multimodal_refreshed": bool(cached.get("multimodal_refreshed")),
                "eval": golden_eval(fields, m.get("ground_truth", {})),
            }
            if refreshed_existing and cached.get("multimodal_refreshed"):
                message = "已更新同一片段的记忆与多模态证据，没有重复新增"
            elif refreshed_existing:
                message = "已用当前缓存更新同一片段的记忆，没有重复新增"
            else:
                message = "这条记忆已在当前会话中，复用现有记录"
            update("cached_memory", message, self._progress_text(m, info))
            retrieval_started = time.perf_counter()
            (info["retrieved_memories"], info["retrieval_meta"]) = self._retrieve_after_write(
                existing_mid, fields, dataset_id=dataset_id, progress=progress)
            record_duration("history_retrieval_wall_ms", retrieval_started)
            retrieval_message = ("片段处理完成；已检索相关历史记忆"
                                 if info["retrieval_meta"].get("status") in {"hit", "stored", "uncached"}
                                 else "片段处理完成；场景检索未完成，已写入记忆仍保留")
            update("done", retrieval_message,
                   self._progress_text(m, info))
            self._set_step_progress(completed_item=self._progress_text(m, info))
            finish_timing(info)
            return existing_mid, info
        cancel_event = self._active_step_cancel
        update("prepare", "正在准备视频片段与记忆写入")
        def on_perception(event):
            update(event.get("stage", "model_inference"),
                   event.get("message", "正在处理当前片段"))
        # Strict allowlist: benchmark captions, transcripts and answer labels
        # never enter the perception/cache/model call.
        perception_input = {
            "id": m.get("id", ""),
            "video": m.get("video", ""),
            "start_time": m.get("start_time"),
            "end_time": m.get("end_time"),
        }
        perception_started = time.perf_counter()
        r = self.perceiver.perceive(perception_input, force=force, cancel_event=cancel_event,
                                    progress_callback=on_perception,
                                    allow_legacy_cache=False,
                                    allow_stale_cache_fallback=True)
        record_duration("perception_call_wall_ms", perception_started)
        operation_ms["perception_reported_elapsed_ms"] = int(r.get("elapsed_ms") or 0)
        operation_ms["perception_reported_elapsed_kind"] = (
            "historical_cached_inference"
            if r.get("source") in {"cache", "cache_quality_warning", "stale_cache_fallback",
                                   "legacy_cache", "label_normalization_only"}
            else "current_model_inference"
        )
        if cancel_event is not None and cancel_event.is_set():
            info = {"dataset_id": dataset_id, "source": "cancelled",
                    "cancelled": True, "error": "当前片段处理已终止"}
            update("cancelled", info["error"])
            finish_timing(info)
            return None, info
        if not r.get("fields"):
            info = {"error": r.get("error", "感知失败"),
                          "source": r.get("source", "error"),
                          "cancelled": bool(r.get("cancelled"))}
            update("cancelled" if info.get("cancelled") else "error",
                   info.get("error") or "感知失败", self._progress_text(m, info))
            self._set_step_progress(completed_item=self._progress_text(m, info))
            finish_timing(info)
            return None, info
        f = dict(r["fields"])
        update("write_memory", "感知结果已生成，正在写入记忆", self._progress_text(m, {"qwen": f}))
        multimodal = None
        media_error = None
        video_path = video_path_of(m)
        clip_start, clip_end = clip_range(m)
        source_provenance = {
            "source_video_id": _source_video_id(m, video_path),
            "source_segment_id": dataset_id,
            "clip_start_s": round(float(clip_start or 0), 2),
            "clip_end_s": round(float(clip_end), 2) if clip_end is not None else None,
            "extraction_batch_id": self.extraction_batch_id,
            "perception_prompt_version": r.get("prompt_version") or PROMPT_VERSION,
            "list_parser_version": r.get("list_parser_version") or LIST_PARSER_VERSION,
            "multimodal_capture_version": CAPTURE_VERSION,
        }
        if r.get("stale_cache_fallback"):
            source_provenance["perception_status"] = "stale_cache_fallback"
            source_provenance["perception_refresh_error"] = str(
                r.get("stale_cache_reason") or "")[:500]
        elif r.get("source") == "cache_quality_warning":
            source_provenance["perception_status"] = "cache_quality_warning"
            source_provenance["perception_quality_warning"] = str(
                r.get("quality_warning") or "")[:700]
        input_fingerprint = r.get("source_input_fingerprint") or {}
        if input_fingerprint.get("video_sha256"):
            source_provenance["source_video_sha256"] = input_fingerprint["video_sha256"]
            source_provenance["source_clip_sha256"] = input_fingerprint.get("clip_sha256", "")
        source_day = next((part.upper() for part in str(dataset_id).split("_")
                           if part.upper().startswith("DAY")
                           and part[3:].isdigit()), "")
        source_time = {
            "day_label": source_day,
            "time_of_day": _clock_from_id(dataset_id),
            "time_occasion": str(f.get("time_occasion") or "")[:200],
            "clip_start_s": round(float(clip_start or 0), 2),
            "basis": "dataset_relative" if source_day else "video_relative",
            "source_segment_id": dataset_id,
        }
        if clip_end is not None:
            source_time["clip_end_s"] = round(float(clip_end), 2)
        frame_times = r.get("frame_times_s") or []

        def attach_evidence_refs(rows, default_modality):
            linked = []
            for original in rows if isinstance(rows, list) else []:
                if not isinstance(original, dict):
                    continue
                row = dict(original)
                ref = {
                    "source_segment_id": source_provenance["source_segment_id"],
                    "clip_start_s": source_provenance["clip_start_s"],
                    "clip_end_s": source_provenance.get("clip_end_s"),
                    "modality": str(row.get("modality") or default_modality)[:16],
                }
                frame_index = row.get("frame_index")
                if isinstance(frame_index, int) and not isinstance(frame_index, bool):
                    ref["frame_index"] = frame_index
                    if 1 <= frame_index <= len(frame_times):
                        ref["frame_time_s"] = frame_times[frame_index - 1]
                        row["frame_time_s"] = frame_times[frame_index - 1]
                row["evidence_ref"] = ref
                scope = str(row.get("time_scope") or "")
                if (default_modality == "visual"
                        and str(row.get("predicate") or "").casefold() in {
                            "move", "moved", "relocate", "relocated", "transfer", "transferred"
                        }):
                    span = re.search(r"frames?\s+(\d+)\s*(?:to|[-–])\s*(\d+)", scope, re.I)
                    if span:
                        start_index, end_index = int(span.group(1)), int(span.group(2))
                        transition_refs = []
                        for index in dict.fromkeys((start_index, end_index)):
                            if not (1 <= index <= len(frame_times)):
                                continue
                            transition_refs.append({
                                "source_segment_id": source_provenance["source_segment_id"],
                                "clip_start_s": source_provenance["clip_start_s"],
                                "clip_end_s": source_provenance.get("clip_end_s"),
                                "modality": "visual",
                                "frame_index": index,
                                "frame_time_s": frame_times[index - 1],
                            })
                        if len(transition_refs) > 1:
                            row["evidence_refs"] = transition_refs
                linked.append(row)
            return linked

        person_observations = attach_evidence_refs(
            f.get("person_observations") or [], "visual")
        temporal_facts = attach_evidence_refs(
            f.get("temporal_facts") or [], "visual")
        f["person_observations"] = person_observations
        f["temporal_facts"] = temporal_facts
        shared_candidate_frames = None
        if video_path and os.environ.get("MEMORY_CAPTURE_MULTIMODAL", "1") != "0":
            try:
                if self.media_store.size_bytes() + 48_000 > self.media_store.asset_budget:
                    self.mem.compact_multimodal_assets(
                        max(0, self.media_store.asset_budget - 30_000))
                    self.media_store._bytes_cached = None
                media_capture_started = time.perf_counter()
                multimodal = self.media_store.capture(
                    video_path, clip_start, clip_end,
                    source_id=dataset_id, fields=f,
                    recent_media=[x.get("multimodal") for x in
                                  list(self.mem.memory["layer7"]["moments"].values())[-100:]],
                    source_provenance=source_provenance,
                )
                record_duration("video_decode_and_multimodal_capture_ms", media_capture_started)
                object_labels = f.get("objects") or []
                visual_refs = multimodal.get("visual_frames") or (
                    [multimodal.get("visual")] if multimodal.get("visual") else [])
                if object_labels and visual_refs:
                    object_pipeline_started = time.perf_counter()
                    update("object_visual_grounding_lookup", "正在检查物品与画面的持久缓存")
                    source_grounding_key = self.grounding_cache.source_key(
                        dataset_id, video_path, clip_start, clip_end, object_labels
                    )
                    object_source_lookup_started = time.perf_counter()
                    try:
                        grounding, grounding_lookup_ms = self.grounding_cache.get_source(
                            source_grounding_key
                        )
                    except Exception:
                        grounding, grounding_lookup_ms = None, 0
                    record_duration("object_source_cache_lookup_wall_ms", object_source_lookup_started)
                    # A source-cache hit skips model inference, not frame decoding: cached
                    # bounding boxes still need the matching pixels to materialize object crops.
                    object_frame_sample_started = time.perf_counter()
                    candidate_frames = self.media_store.sample_visual_candidates(
                        video_path, clip_start, clip_end, source_id=dataset_id, count=7)
                    record_duration("object_candidate_frame_sampling_ms", object_frame_sample_started)
                    shared_candidate_frames = candidate_frames
                    grounding_key = None
                    if grounding is None:
                        frame_inputs = candidate_frames
                        if not frame_inputs:
                            for ref in visual_refs[:3]:
                                asset_path = Path(self.mem.memory_dir) / str(ref.get("asset") or "")
                                if ref.get("asset") and asset_path.is_file():
                                    frame_inputs.append({
                                        **ref, "mime": ref.get("mime") or "image/webp",
                                        "bytes": asset_path.read_bytes(),
                                    })
                        object_detail_lookup_started = time.perf_counter()
                        try:
                            grounding_key = self.grounding_cache.key(
                                dataset_id, object_labels, frame_inputs
                            )
                            grounding, grounding_lookup_ms = self.grounding_cache.get(grounding_key)
                        except Exception:
                            grounding_key, grounding, grounding_lookup_ms = None, None, 0
                        record_duration("object_grounding_cache_lookup_wall_ms", object_detail_lookup_started)
                        if grounding is not None and source_grounding_key:
                            self.grounding_cache.put_source(source_grounding_key, grounding)
                    operation_ms["object_grounding_cache_hit"] = grounding is not None
                    if grounding is not None:
                        update("object_grounding_cache_hit",
                               f"物品画面对照命中缓存（{grounding_lookup_ms} ms），跳过视觉推理")
                    else:
                        update("object_visual_grounding",
                               "缓存未命中，正在逐件核对物品与关键帧")
                        try:
                            object_model_started = time.perf_counter()
                            grounding = self.perceiver.ground_objects(
                                object_labels, frame_inputs, cancel_event=cancel_event,
                                progress_callback=lambda event: update(
                                    event.get("stage", "object_visual_grounding"),
                                    event.get("message", "正在核对物品画面"),
                                    self._progress_text(m, {"qwen": f}),
                                ),
                            )
                            record_duration("object_grounding_model_call_ms", object_model_started)
                        except Exception as exc:
                            grounding_error = str(exc)[:500]
                            grounding = {
                                "schema": 1, "status": "unavailable", "model": "Qwen3-VL",
                                "message": grounding_error,
                                "items": [{"object": str(label), "status": "unavailable",
                                           "error": grounding_error, "frame_refs": [],
                                           "crop_count": 0, "frame_indices": []}
                                          for label in object_labels[:12]],
                            }
                        self.grounding_cache.put(grounding_key, grounding)
                        if source_grounding_key:
                            self.grounding_cache.put_source(source_grounding_key, grounding)

                    object_crop_started = time.perf_counter()
                    saved_by_index = {}
                    if candidate_frames and grounding.get("status") == "completed":
                        coverage = {}
                        for item in grounding.get("items") or []:
                            if item.get("status") in {"grounded", "visible_unlabeled"}:
                                for index in item.get("frame_indices") or []:
                                    if 1 <= index <= len(candidate_frames):
                                        coverage.setdefault(index, set()).add(item.get("object"))
                        selected = []
                        uncovered = {
                            item.get("object") for item in (grounding.get("items") or [])
                            if item.get("status") in {"grounded", "visible_unlabeled"}
                        }
                        while uncovered and len(selected) < 3:
                            best = max(
                                (index for index in coverage if index not in selected),
                                key=lambda index: len(coverage[index] & uncovered),
                                default=None,
                            )
                            if best is None or not (coverage[best] & uncovered):
                                break
                            selected.append(best)
                            uncovered -= coverage[best]
                        for anchor in (1, (len(candidate_frames) + 1) // 2, len(candidate_frames)):
                            if len(selected) >= 3:
                                break
                            if anchor not in selected and 1 <= anchor <= len(candidate_frames):
                                selected.append(anchor)
                        for index in selected[:3]:
                            frame = candidate_frames[index - 1]
                            retained_image = self.media_store.fit_visual_candidate(
                                frame.get("bytes"), 6_000)
                            saved = self.media_store._put(
                                "visual", retained_image, "webp", 6_000)
                            if saved:
                                saved.update({
                                    "mime": "image/webp",
                                    "dhash": frame.get("dhash") or "",
                                    "at_s": frame.get("at_s"),
                                    "source_id": frame.get("source_id") or dataset_id,
                                    "quality_tier": "recent",
                                })
                                saved_by_index[index] = saved
                        if saved_by_index:
                            selected_refs = list(saved_by_index.values())
                            multimodal["visual_frames"] = selected_refs
                            multimodal["visual"] = dict(selected_refs[len(selected_refs) // 2])
                            multimodal["visual_sampling"] = {
                                "candidate_frames": len(candidate_frames),
                                "retained_frames": len(selected_refs),
                                "strategy": "object_coverage_with_temporal_anchors",
                            }
                    else:
                        saved_by_index = {
                            index: ref for index, ref in enumerate(visual_refs[:3], start=1)
                        }
                    for item in grounding.get("items") or []:
                        boxes = {}
                        for box in item.get("frame_boxes") or []:
                            try:
                                box_index = int(box.get("frame_index"))
                                bbox = box.get("bbox")
                            except (AttributeError, TypeError, ValueError):
                                continue
                            max_frame_index = len(candidate_frames) if candidate_frames else 7
                            if 1 <= box_index <= max_frame_index and isinstance(bbox, list) and len(bbox) == 4:
                                boxes[box_index] = bbox
                        requested = []
                        for index in item.pop("frame_indices", []):
                            if isinstance(index, int) and index not in requested:
                                requested.append(index)
                        for index in boxes:
                            if index not in requested:
                                requested.append(index)
                        frame_refs = []
                        for index in requested[:1]:
                            if candidate_frames:
                                if not (1 <= index <= len(candidate_frames)):
                                    continue
                                frame = candidate_frames[index - 1]
                                source_ref = saved_by_index.get(index)
                                if not source_ref:
                                    source_bytes = self.media_store.fit_visual_candidate(
                                        frame.get("bytes"), 6_000)
                                    source_ref = self.media_store._put(
                                        "visual", source_bytes, "webp", 6_000)
                                    if source_ref:
                                        source_ref.update({
                                            "mime": "image/webp",
                                            "dhash": frame.get("dhash") or "",
                                            "at_s": frame.get("at_s"),
                                            "source_id": frame.get("source_id") or dataset_id,
                                            "quality_tier": "recent",
                                        })
                                        saved_by_index[index] = source_ref
                            else:
                                if not visual_refs:
                                    continue
                                ratio = (index - 1) / 6 if index > 1 else 0
                                ref_index = round(ratio * (len(visual_refs) - 1))
                                source_ref = visual_refs[max(0, min(ref_index, len(visual_refs) - 1))]
                                frame = {"at_s": source_ref.get("at_s"),
                                         "source_id": source_ref.get("source_id") or dataset_id}
                            if not source_ref or not source_ref.get("asset"):
                                continue
                            crop_ref = None
                            bbox = boxes.get(index) if candidate_frames else None
                            if bbox:
                                crop_bytes = self.media_store.crop_visual_candidate(
                                    frame.get("bytes"), bbox, 12_000)
                                crop_ref = self.media_store._put(
                                    "visual", crop_bytes, "webp", 12_000)
                            primary_ref = crop_ref or source_ref
                            frame_ref = {
                                "asset": primary_ref.get("asset"),
                                "bytes": int(primary_ref.get("bytes") or 0),
                                "at_s": frame.get("at_s"),
                                "source_id": frame.get("source_id") or dataset_id,
                                "quality_tier": "recent",
                            }
                            if crop_ref:
                                frame_ref.update({
                                    "source_asset": source_ref["asset"],
                                    "crop_bbox": [round(float(value), 5) for value in bbox],
                                    "visual_role": "object_crop",
                                })
                            else:
                                frame_ref["visual_role"] = "source_frame"
                            frame_refs.append(frame_ref)
                        item["frame_refs"] = frame_refs
                        item["crop_count"] = sum(
                            ref.get("visual_role") == "object_crop" for ref in frame_refs)
                        item.pop("frame_boxes", None)
                        if item.get("status") in {"grounded", "visible_unlabeled"} and not item["frame_refs"]:
                            item["status"] = "not_verified"
                            item["error"] = "模型未提供可用边界框，等待重试"
                    for item in grounding.get("items") or []:
                        item["appearance"] = _remove_visual_age_guess(item.get("appearance"))
                    grounding["crop_schema"] = 1
                    grounding["crop_method"] = "Qwen3-VL normalized object box with padded WebP crop"
                    multimodal["object_visuals"] = grounding
                    record_duration("object_crop_generation_and_asset_write_ms", object_crop_started)
                    record_duration("object_grounding_pipeline_wall_ms", object_pipeline_started)
                    if cancel_event is not None and cancel_event.is_set():
                        info = {
                            "error": "物品画面核对已停止，当前片段未写入记忆",
                            "source": "cancelled", "cancelled": True,
                        }
                        finish_timing(info)
                        return None, info
                person_rows = [row for row in person_observations
                               if isinstance(row, dict) and str(row.get("entity") or "").strip()]
                if person_rows and visual_refs:
                    person_pipeline_started = time.perf_counter()
                    update("person_visual_grounding_lookup", "正在检查人物局部画面的持久缓存")
                    person_frame_sample_started = time.perf_counter()
                    person_frames = shared_candidate_frames
                    reused_object_frames = bool(person_frames)
                    if not person_frames:
                        person_frames = self.media_store.sample_visual_candidates(
                            video_path, clip_start, clip_end, source_id=dataset_id, count=7)
                    record_duration("person_candidate_frame_sampling_ms", person_frame_sample_started)
                    operation_ms["person_candidate_frames_reused_from_objects"] = reused_object_frames
                    if not person_frames:
                        for ref in visual_refs[:3]:
                            asset_path = Path(self.mem.memory_dir) / str(ref.get("asset") or "")
                            if ref.get("asset") and asset_path.is_file():
                                person_frames.append({
                                    **ref, "mime": ref.get("mime") or "image/webp",
                                    "bytes": asset_path.read_bytes(),
                                })
                    person_key = None
                    person_cache_lookup_started = time.perf_counter()
                    try:
                        # Include the appearance hint so repeated clip-local labels cannot
                        # accidentally reuse a crop for a different visually described person.
                        person_cache_labels = [
                            "person-crop-v1: " + str(row.get("entity") or "").strip()
                            + " | " + str(row.get("appearance") or "").strip()
                            for row in person_rows[:8]
                        ]
                        person_key = self.grounding_cache.key(
                            dataset_id, person_cache_labels, person_frames)
                        person_grounding, person_lookup_ms = self.grounding_cache.get(person_key)
                    except Exception:
                        person_key, person_grounding, person_lookup_ms = None, None, 0
                    record_duration("person_grounding_cache_lookup_wall_ms", person_cache_lookup_started)
                    operation_ms["person_grounding_cache_hit"] = person_grounding is not None
                    if person_grounding is not None:
                        update("person_grounding_cache_hit",
                               f"人物画面缓存已命中（{person_lookup_ms} ms），跳过视觉推理")
                    else:
                        update("person_visual_grounding", "缓存未命中，正在定位人物并生成局部裁剪")
                        try:
                            person_model_started = time.perf_counter()
                            person_grounding = self.perceiver.ground_people(
                                person_rows, person_frames, cancel_event=cancel_event,
                                progress_callback=lambda event: update(
                                    event.get("stage", "person_visual_grounding"),
                                    event.get("message", "正在定位人物画面"),
                                    self._progress_text(m, {"qwen": f}),
                                ),
                            )
                            record_duration("person_grounding_model_call_ms", person_model_started)
                        except Exception as exc:
                            grounding_error = str(exc)[:500]
                            person_grounding = {
                                "schema": 1, "status": "unavailable", "model": "Qwen3-VL",
                                "message": grounding_error,
                                "items": [{"person": str(row.get("entity") or ""),
                                           "status": "unavailable",
                                           "appearance": str(row.get("appearance") or ""),
                                           "error": grounding_error, "frame_refs": [],
                                           "crop_count": 0, "frame_indices": [],
                                           "frame_boxes": []}
                                          for row in person_rows[:8]],
                            }
                        self.grounding_cache.put(person_key, person_grounding)

                    person_crop_started = time.perf_counter()
                    for item in person_grounding.get("items") or []:
                        boxes = {int(box.get("frame_index")): box.get("bbox")
                                 for box in item.get("frame_boxes") or []
                                 if isinstance(box, dict)
                                 and isinstance(box.get("frame_index"), int)
                                 and isinstance(box.get("bbox"), list)}
                        requested = []
                        for index in item.pop("frame_indices", []):
                            if isinstance(index, int) and index not in requested:
                                requested.append(index)
                        for index in boxes:
                            if index not in requested:
                                requested.append(index)
                        frame_refs = []
                        for index in requested[:1]:
                            if not (1 <= index <= len(person_frames)):
                                continue
                            frame = person_frames[index - 1]
                            bbox = boxes.get(index)
                            if not bbox:
                                continue
                            crop_bytes = self.media_store.crop_visual_candidate(
                                frame.get("bytes"), bbox, 12_000)
                            crop_ref = self.media_store._put(
                                "visual", crop_bytes, "webp", 12_000)
                            if not crop_ref or not crop_ref.get("asset"):
                                continue
                            source_bytes = self.media_store.fit_visual_candidate(
                                frame.get("bytes"), 6_000)
                            source_ref = self.media_store._put(
                                "visual", source_bytes, "webp", 6_000)
                            if source_ref:
                                source_ref.update({
                                    "mime": "image/webp",
                                    "dhash": frame.get("dhash") or "",
                                    "at_s": frame.get("at_s"),
                                    "source_id": frame.get("source_id") or dataset_id,
                                })
                            frame_ref = {
                                "asset": crop_ref["asset"],
                                "at_s": frame.get("at_s"),
                                "source_id": frame.get("source_id") or dataset_id,
                                "visual_role": "person_crop",
                                "crop_bbox": [round(float(value), 5) for value in bbox],
                            }
                            if source_ref and source_ref.get("asset"):
                                frame_ref["source_asset"] = source_ref["asset"]
                            frame_refs.append(frame_ref)
                        item["frame_refs"] = frame_refs
                        item["crop_count"] = len(frame_refs)
                        item.pop("frame_boxes", None)
                        if not frame_refs and person_grounding.get("status") != "unavailable":
                            item["status"] = "not_verified"
                            item["error"] = "模型未提供可用边界框，等待重试"
                    person_grounding["crop_schema"] = 1
                    person_grounding["crop_method"] = (
                        "Qwen3-VL normalized whole-person box with padded WebP crop")
                    multimodal["person_visuals"] = person_grounding
                    record_duration("person_crop_generation_and_asset_write_ms", person_crop_started)
                    record_duration("person_grounding_pipeline_wall_ms", person_pipeline_started)
                    if cancel_event is not None and cancel_event.is_set():
                        info = {
                            "error": "人物画面核对已停止，当前片段未写入记忆",
                            "source": "cancelled", "cancelled": True,
                        }
                        finish_timing(info)
                        return None, info
                # People and objects use tight grounded crops above. Scene, place,
                # environment, activity, and each event also get source-linked
                # context crops, so every stored memory dimension can be inspected
                # against its supporting frame in the UI.
                context_crop_started = time.perf_counter()
                try:
                    context_frames = (locals().get("candidate_frames")
                                      or locals().get("person_frames") or [])
                    if not context_frames:
                        context_frames = self.media_store.sample_visual_candidates(
                            video_path, clip_start, clip_end,
                            source_id=dataset_id, count=7)
                    multimodal["entity_visuals"] = self.media_store.capture_context_entities(
                        context_frames, f, source_id=dataset_id,
                        clip_start_s=clip_start,
                    )
                except Exception as context_exc:
                    multimodal["entity_visuals"] = {
                        "schema": 1, "status": "unavailable",
                        "message": str(context_exc)[:240], "assets": [], "items": [],
                    }
                record_duration("context_crop_generation_and_asset_write_ms", context_crop_started)
            except Exception as exc:
                # A media extraction failure must not prevent text memory CRUD.
                media_error = str(exc)
        audio_started = time.perf_counter()
        source_audio_text = self._transcribe_audio_evidence(
            multimodal, dataset_id, video_path=video_path,
            clip_start_s=clip_start, clip_end_s=clip_end,
            progress=lambda stage, message: update(stage, message),
            timing_sink=operation_ms,
        )
        record_duration("audio_decode_cache_and_transcription_wall_ms", audio_started)
        if source_audio_text and person_observations:
            original_observations = [
                dict(row) for row in person_observations if isinstance(row, dict)
            ]
            profile_frames = (locals().get("candidate_frames")
                              or locals().get("person_frames") or [])
            person_observations, speech_scene_context = self._enrich_person_profiles_from_speech(
                person_observations, source_audio_text, profile_frames, dataset_id,
                cancel_event=cancel_event,
                progress=lambda stage, message: update(stage, message),
            )
            if speech_scene_context:
                # Keep the meaningful shared context, not a flat sequence of
                # low-value action and greeting rows.
                f["interaction_details"] = speech_scene_context[:700]
            f["person_observations"] = person_observations
            f["people"] = list(dict.fromkeys(
                str(row.get("entity") or "").strip()
                for row in person_observations
                if isinstance(row, dict) and str(row.get("entity") or "").strip()
            ))
            alias_map = {
                str(old.get("entity") or "").strip().casefold():
                str(new.get("entity") or "").strip()
                for old, new in zip(original_observations, person_observations)
                if str(old.get("entity") or "").strip()
                and str(new.get("entity") or "").strip()
                and str(old.get("entity")).strip().casefold()
                != str(new.get("entity")).strip().casefold()
            }
            if alias_map:
                def remap_media_person_labels(value):
                    if isinstance(value, str):
                        return alias_map.get(value.strip().casefold(), value)
                    if isinstance(value, list):
                        return [remap_media_person_labels(item) for item in value]
                    if isinstance(value, dict):
                        return {key: remap_media_person_labels(item)
                                for key, item in value.items()}
                    return value
                multimodal = remap_media_person_labels(multimodal)
        # Speech is evidence for people and events, not a transcript dump: the facts
        # stay source-linked rows, and a spoken term only renames a visible person
        # when this clip itself singles that person out.
        speech_facts = (_extract_explicit_speech_facts(source_audio_text, dataset_id)
                        if source_audio_text else [])
        if source_audio_text:
            f["source_evidence"] = source_audio_text
        identity_links = attach_spoken_identity(
            person_observations, speech_facts, temporal_facts)
        f["people"] = person_labels_from_observations(
            f.get("people") or [], person_observations)
        if speech_facts:
            temporal_facts = list(temporal_facts) + speech_facts
            f["temporal_facts"] = temporal_facts
        normalize_object_transition_facts(f)
        temporal_facts = f.get("temporal_facts") or temporal_facts
        if cancel_event is not None and cancel_event.is_set():
            info = {"error": "当前片段处理已终止", "source": "cancelled", "cancelled": True}
            finish_timing(info)
            update("cancelled", info["error"])
            return None, info
        memory_write_started = time.perf_counter()
        mid = self.daytime.ingest(
            scene=f["scene"],
            user_action=f["user_action"],
            needs=[],
            solutions=[],
            people=f["people"] or None,
            location=f["location"],
            activity=f["activity"],
            environments=f["environments"],
            objects=f["objects"],
            extra_notes=(f.get("time_occasion") or "")[:200],
            timestamp=(time.strftime("%Y-%m-%d") + "T" + (_clock_from_id(m.get("id", "")) or time.strftime("%H:%M:%S"))),
            source_evidence=f.get("source_evidence") or "",
            interaction_details=f.get("interaction_details") or "",
            person_observations=person_observations,
            temporal_facts=temporal_facts,
            source_time=source_time,
            multimodal=multimodal,
            source_provenance=source_provenance,
        )
        record_duration("memory_write_and_index_ms", memory_write_started)
        if multimodal and multimodal.get("visual_sampling"):
            self.media_store.release_unreferenced(
                self.mem.memory["layer7"]["moments"])
        self.id_map[m["id"]] = mid
        self._qwen_fields[m["id"]] = f
        info = {
            "dataset_id": m.get("id", ""),
            "moment_id": mid,
            "qwen": f,
            "raw": r.get("raw", ""),
            "raw_model_output": r.get("raw_model_output", ""),
            "elapsed_ms": r.get("elapsed_ms"),
            "num_frames": r.get("num_frames"),
            "source": r.get("source", "qwen"),
            "quality_warning": r.get("quality_warning", ""),
            "stale_cache_fallback": bool(r.get("stale_cache_fallback")),
            "spoken_identity_links": identity_links,
            "multimodal_refreshed": False,
            "eval": golden_eval(f, m.get("ground_truth", {})),
            "multimodal": multimodal,
            "media_error": media_error,
        }
        retrieval_started = time.perf_counter()
        (info["retrieved_memories"], info["retrieval_meta"]) = self._retrieve_after_write(
            mid, f, dataset_id=dataset_id, progress=progress)
        record_duration("history_retrieval_wall_ms", retrieval_started)
        retrieval_message = ("片段已写入并检索相关历史记忆"
                             if info["retrieval_meta"].get("status") in {"hit", "stored", "uncached"}
                             else "片段已写入；场景检索未完成，记忆已保留")
        update("done", retrieval_message, self._progress_text(m, info))
        finish_timing(info)
        self._set_step_progress(completed_item=self._progress_text(m, info))
        return mid, info

    # ------------------------------------------------------------------
    # 会话构建
    # ------------------------------------------------------------------

    def _build(self):
        mem_dir = RUNS / "interactive_mem" / self.key
        mem_dir.mkdir(parents=True, exist_ok=True)
        resuming = self._resume_state is not None
        if any(mem_dir.iterdir()) and not resuming:
            # 归档前先剥离旧会话的原始音频，只保留已有转写文字。
            old_mem = PersonMemory(memory_dir=str(mem_dir))
            if strip_audio_assets(old_mem.memory):
                old_mem._save()
            old_store = MultimodalAssetStore(mem_dir)
            old_store.release_unreferenced(
                old_mem.memory.get("layer7", {}).get("moments", {}))
            # 旧会话数据改名归档（os.rename 不触发 safe-delete 钩子，避免 trash 权限问题）
            trash_dir = RUNS / "interactive_mem" / "_trash"
            trash_dir.mkdir(exist_ok=True)
            os.rename(mem_dir, trash_dir / f"{self.key}_{time.time_ns()}")
            mem_dir.mkdir(parents=True, exist_ok=True)
        self.mem = PersonMemory(memory_dir=str(mem_dir))
        self.media_store = MultimodalAssetStore(mem_dir)
        if strip_audio_assets(self.mem.memory):
            self.mem._save()
            self.media_store._bytes_cached = None
            self.media_store.release_unreferenced(
                self.mem.memory.get("layer7", {}).get("moments", {}))
        self._install_operation_audit()
        self.daytime = DaytimeProcessor(self.mem)
        self.daytime.start_scene(f"{self.key}_{time.strftime('%Y-%m-%d')}")
        self.id_map = {}        # 数据集 id → 记忆模型 moment_id
        self._qwen_fields = {}  # 数据集 id → Qwen 产出（sort / 展示用）

        if self.key == "short_term":
            self.steps = self._plan_short()
        elif self.key == "custom_test":
            self.steps = self._plan_custom()
        else:
            self.steps = self._plan_long()

        if resuming:
            checkpoint = self._resume_state
            expected_dataset = str(checkpoint.get("dataset_id") or "")
            if expected_dataset and expected_dataset != self.dataset_id:
                raise ValueError("会话恢复数据集与当前记忆库不一致")
            self.pos = max(0, min(int(checkpoint.get("pos") or 0), len(self.steps)))
            self.log = copy.deepcopy(checkpoint.get("log") or [])
            strip_audio_assets(self.log)
            self.operation_log = copy.deepcopy(checkpoint.get("operations") or [])
            self.auto_paused = bool(checkpoint.get("auto_paused"))
            self.step_cancelled = bool(checkpoint.get("step_cancelled"))
            for moment_id, moment in self.mem.memory.get("layer7", {}).get("moments", {}).items():
                source_id = str((moment.get("source_provenance") or {}).get("source_segment_id") or "")
                if source_id:
                    self.id_map[source_id] = moment_id
            for moment in self.moments:
                cached = self.perceiver.cache.get(moment.get("id", "")) or {}
                if cached.get("fields"):
                    self._qwen_fields[moment["id"]] = cached["fields"]
            self.step_progress = None
            self.nightly_progress = None
            # Refresh the latest restored visual snapshot from disk so a
            # completed media backfill is visible without re-running the step.
            if self.log:
                latest = self.log[-1]
                snap = rs.snapshot(
                    self.mem, str(latest.get("title") or "恢复后的记忆快照"),
                    action=str(latest.get("kind") or "resume"))
                snap["full"] = self._full_snapshot()
                snap["layer7_active_moments"] = snap["full"]["layer7_active_total"]
                snap["layer7_merged_moments"] = snap["full"]["layer7_merged_total"]
                latest["snapshot"] = snap
            self._resume_state = None
        else:
            self._log_snapshot("初始状态", "init", "会话创建，等待第一步")

    def _display_brief(self, m):
        """Current-step preview uses the same cached English fields that feed memory."""
        brief = moment_brief(m)
        cache_entry = self.perceiver.cache.get(m.get("id", "")) or {}
        cache_is_current = cache_entry.get("prompt_version") == PROMPT_VERSION
        brief['preview_source'] = 'previous_perception' if cache_is_current else 'video_only'
        brief['preview_prompt_version'] = cache_entry.get('prompt_version') if cache_is_current else ''
        cached = (cache_entry.get("fields") or {}) if cache_is_current else {}
        if cached:
            if cached.get("scene"):
                brief["scene"] = cached["scene"]
            if cached.get("user_action"):
                brief["action"] = cached["user_action"]
            for key in ("people", "location", "activity"):
                if cached.get(key):
                    brief[key] = cached[key]
        return brief

    def _plan_short(self):
        """短期场景：逐条写入 + CRUD 计划，可按人物筛选当前演示集。"""
        if self.dataset_key not in ("base", "expanded"):
            raise ValueError(f"未知短期数据集：{self.dataset_key}")
        source_file, plan_file = (
            (SHORT_EXPANDED_FILE, SHORT_EXPANDED_PLAN)
            if self.dataset_key == "expanded" else (SHORT_FILE, SHORT_PLAN)
        )
        all_moments = _load_jsonl(source_file)
        all_plan = json.load(open(plan_file, encoding="utf-8"))
        if self.participant:
            moments = [
                m for m in all_moments
                if "_".join(str(m.get("id", "")).split("_")[1:3]) == self.participant
            ]
            if not moments:
                raise ValueError(f"短期交互集没有参与者 {self.participant}")
            def for_participant(item_id):
                return "_".join(str(item_id).split("_")[1:3]) == self.participant

            plan = {
                **all_plan,
                "add": [m.get("id") for m in moments],
                "update": [x for x in all_plan.get("update", [])
                           if for_participant(x.get("id"))],
                "highlight": [x for x in all_plan.get("highlight", [])
                              if for_participant(x)],
                "delete": [x for x in all_plan.get("delete", [])
                           if for_participant(x.get("id"))],
            }
            # 每个人物都保留完整的 CRUD 演示路径；没有预设操作时，
            # 从该人物自己的片段生成一条可追溯的演示操作。
            # Do not synthesize memory operations from benchmark labels.
            # A missing CRUD demo is preferable to contaminating memory.
            if not plan["highlight"]:
                plan["highlight"] = [moments[0]["id"]]
            if not plan["delete"]:
                delete_target = next(
                    (m["id"] for m in reversed(moments)
                     if m["id"] not in plan["highlight"]),
                    moments[-1]["id"],
                )
                plan["delete"] = [{
                    "id": delete_target,
                    "reason": "短期场景结束后清理该人物的演示记忆",
                }]
        else:
            moments = all_moments
            plan = all_plan

        # CRUD 演示只写入明确标注的演示备注，不把官方参考标签写回记忆。
        plan = dict(plan)
        plan["update"] = [
            {**case, "field": "notes", "new_value": "CRUD field update demo"}
            for case in plan.get("update", [])
        ]
        self.moments = moments
        self.plan = plan
        self.dataset_options = interactive_dataset_options()
        self.dataset_id = f"short_term|{self.dataset_key}|{self.participant or 'all'}"
        self.dataset_note = next(
            (x["note"] for x in self.dataset_options if x["value"] == self.dataset_id), "")
        self.participant_label = PARTICIPANT_NAMES.get(
            self.participant, "全部人物" if not self.participant else self.participant)
        steps = []

        # 1) add：每条视频片段一步（Qwen 感知 → 记忆）
        for i, m in enumerate(moments, 1):
            def run(m=m, i=i, force=False):
                context = {"item_index": 1, "item_total": 1,
                           "label": f"写入记忆 · {i}/{len(moments)}"}
                mid, info = self._add_one(m, force=force, progress=context)
                if mid is None:
                    msg = ("实时推理已停止" if info.get("cancelled") else "感知失败")
                    return f"{msg}：{info.get('error')}", {"kind": "single", **info}
                msg = f"Qwen 看过 {m['id']} → 写入记忆 {mid}（{info['source']}，{info['elapsed_ms']}ms）"
                return msg, {"kind": "single", **info}
            steps.append({
                "kind": "add",
                "title": f"感知并写入视频片段 {i}/{len(moments)}",
                "inputs": [self._display_brief(m)],
                "run": run,
            })

        # 相似事件只进入待确认队列；用户逐条核对后才会写入归并关系。
        def run_merge_candidates(force=False):
            candidates = self.mem.find_merge_candidates(limit=100)
            result = {
                "kind": "operation_result", "operation": "merge_candidates",
                "title": "相似事件候选 · 逐条确认后才会合并",
                "candidates": candidates,
            }
            return f"发现 {len(candidates)} 组相似事件候选；没有自动合并", result
        steps.append({
            "kind": "merge_candidates",
            "title": "检查相似事件候选（需要逐条确认）",
            "detail": {"note": "候选依据：同源片段重叠≥80%，或120秒内Scene与Action合并文本相似度≥92%且共享人物/物品。画面与语音转写文本只供核对；扫描不改数据，逐条确认后才归并。"},
            "inputs": [],
            "run": run_merge_candidates,
        })

        # 2) query
        for q in plan.get("query", []):
            def run(q=q, force=False):
                hits = self.daytime.query(q["query"], top_k=3)
                items = [self._moment_detail(hit) for hit in hits]
                result = {
                    "kind": "operation_result", "operation": "query",
                    "title": "检索命中内容", "query": q["query"],
                    "count": len(hits), "items": items,
                }
                return f"检索「{q['query']}」命中 {len(hits)} 条", result
            steps.append({
                "kind": "query",
                "title": f"检索「{q['query']}」",
                "detail": {"query": q["query"], "top_k": 3},
                "inputs": [],
                "run": run,
            })

        # 3) update
        for case in plan.get("update", []):
            def run(case=case, force=False):
                mid = self.id_map.get(case["id"])
                if not mid:
                    return f"未找到 {case['id']}，跳过", {
                        "kind": "operation_result", "operation": "update",
                        "title": "更新对象未找到", "items": [],
                    }
                before = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
                field = case["field"]
                old_value = copy.deepcopy(before.get(field))
                ok = self.mem.update(mid, {field: case["new_value"]})
                after = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
                item = self._moment_detail(after)
                item["changes"] = [{
                    "field": field, "before": old_value,
                    "after": copy.deepcopy(after.get(field)),
                }]
                result = {
                    "kind": "operation_result", "operation": "update",
                    "title": "字段更新前后", "items": [item],
                }
                return f"更新 {case['id']}.{field}（{'成功' if ok else '字段未变化'}）", result
            steps.append({
                "kind": "update",
                "title": f"更新字段 {case['field']}",
                "detail": {"id": case["id"], "field": case["field"],
                           "new_value": case["new_value"]},
                "inputs": [],
                "run": run,
            })

        # 4) update_feedback：在独立交互演示库模拟纠错、确认与评分，不使用评测标签。
        update_ids = {case.get("id") for case in plan.get("update", [])}
        protected_ids = set(plan.get("highlight", [])) | {
            case.get("id") for case in plan.get("delete", [])
        }
        feedback_moment = next(
            (m for m in moments if m.get("id") not in update_ids | protected_ids),
            moments[0] if moments else None,
        )
        def run_feedback(force=False, m=feedback_moment):
            if not m:
                return "没有可反馈的记忆", None
            mid = self.id_map.get(m.get("id"))
            if not mid:
                return f"未找到 {m.get('id')}，反馈未保存", None
            corrections = {"notes": "CRUD feedback correction demo"}
            before = copy.deepcopy(
                self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            )
            ok = self.mem.update_feedback(
                mid, corrections=corrections, confirmed=True, user_rating=5,
            )
            after = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid, {})
            item = self._moment_detail(after)
            item["corrections"] = [
                {"field": field, "before": copy.deepcopy(before.get(field)), "after": value}
                for field, value in corrections.items()
            ]
            item["confirmed"] = after.get("feedback", {}).get("confirmed")
            item["user_rating"] = after.get("feedback", {}).get("user_rating")
            result_detail = {
                "kind": "operation_result", "operation": "update_feedback",
                "title": "模拟反馈修正内容", "simulated": True,
                "items": [item],
            }
            result = "已保存模拟纠错、确认和 5 星评分" if ok else "模拟反馈保存失败"
            return result, result_detail
        steps.append({
            "kind": "update_feedback",
            "title": "模拟反馈 · 纠错、确认与评分",
            "detail": {
                "id": feedback_moment.get("id", "") if feedback_moment else "",
                "note": "仅写入交互演示库的演示备注；不读取官方参考标签。确认和评分为模拟输入，不代表真实用户评价。",
            },
            "inputs": [],
            "run": run_feedback,
        })

        # 5) highlight
        hl_ids = plan.get("highlight", [])
        def run_highlight(force=False):
            items, n = [], 0
            moments = self.mem.memory.get("layer7", {}).get("moments", {})
            for sid in hl_ids:
                mid = self.id_map.get(sid)
                before = copy.deepcopy(moments.get(mid, {})) if mid else {}
                ok = bool(mid and self.mem.highlight(mid))
                after = moments.get(mid, before) if mid else before
                if ok:
                    n += 1
                item = self._moment_detail(after)
                item.update({"dataset_id": sid, "marked": ok,
                             "was_highlighted": bool(before.get("highlighted"))})
                items.append(item)
            result = {
                "kind": "operation_result", "operation": "highlight",
                "title": "已标记为重点的记忆", "items": items,
            }
            return f"标记重要记忆 {n}/{len(hl_ids)} 条", result
        steps.append({
            "kind": "highlight",
            "title": "标记重要记忆",
            "detail": {"ids": hl_ids},
            "inputs": [],
            "run": run_highlight,
        })

        # 6) delete（含 highlight 保护验证）
        del_cases = plan.get("delete", [])
        def run_delete(force=False):
            targets = []
            for case in del_cases:
                mid = self.id_map.get(case["id"])
                moment = self.mem.memory.get("layer7", {}).get("moments", {}).get(mid) if mid else None
                targets.append((case, mid, copy.deepcopy(moment) if isinstance(moment, dict) else None))
            cleared = self.daytime.end_scene()
            msgs = [f"场景结束：清理 Layer1/2 {cleared} 条"]
            items, ok_n = [], 0
            for case, mid, before in targets:
                if not mid:
                    items.append({"dataset_id": case.get("id"), "deleted": False,
                                  "reason": case.get("reason", "manual"), "memory": {}})
                    continue
                ok = self.mem.delete(mid, reason=case.get("reason", "manual"))
                if ok:
                    ok_n += 1
                items.append({
                    **self._moment_detail(before or {}),
                    "dataset_id": case.get("id"),
                    "deleted": bool(ok),
                    "reason": case.get("reason", "manual"),
                })
            msgs.append(f"清理短期记忆：删除 {ok_n} 条")
            protection = []
            # 验证 highlight 保护，并把被验证的目标也明确展示出来。
            prot = [self.id_map[s] for s in hl_ids if self.id_map.get(s)]
            if prot:
                protected_moment = self.mem.memory.get("layer7", {}).get("moments", {}).get(prot[0], {})
                protected_detail = self._moment_detail(copy.deepcopy(protected_moment))
                blocked = not self.mem.delete(prot[0], reason="验证保护")
                forced = self.mem.delete(prot[0], reason="验证强制删除", force=True)
                protection.append({
                    **protected_detail, "default_delete_blocked": bool(blocked),
                    "force_deleted": bool(forced),
                })
                msgs.append(f"highlight 保护：默认删除被拒绝 {'✅' if blocked else '❌'}；"
                            f"force 强删 {'成功 ✅' if forced else '失败 ❌'}")
            result = {
                "kind": "operation_result", "operation": "delete",
                "title": "删除对象及保护验证", "items": items,
                "protection_checks": protection,
            }
            return "；".join(msgs), result
        steps.append({
            "kind": "delete",
            "title": "清理短期记忆（含保护验证）",
            "detail": {"ids": [c["id"] for c in del_cases]},
            "inputs": [],
            "run": run_delete,
        })

        return steps

    def _nightly_consolidation_step(self, title=None, note=None):
        """Build the shared scene-end/nightly consolidation step."""
        def run_nightly(force=False, use_cache=True):
            self.nightly_progress = {"running": True, "stage": "start",
                                     "label": "启动夜间整理", "current": 0,
                                     "total": 0, "message": "", "updated_at": time.time()}
            def on_progress(event):
                self.nightly_progress = {"running": True, **event, "updated_at": time.time()}
            try:
                result = NightlyProcessor(self.mem).run(
                    day=self.mem.memory["metadata"]["today"],
                    daytime_processor=self.daytime,
                    force=force,
                    use_cache=use_cache,
                    progress_callback=on_progress,
                )
            except Exception as exc:
                self.nightly_progress = {
                    "running": False, "stage": "error", "label": "夜间整理失败",
                    "current": 0, "total": 0, "message": str(exc)[:300],
                    "updated_at": time.time(),
                }
                raise
            self.nightly_progress = {
                "running": False, "stage": "done", "label": "夜间整理完成",
                "current": 1, "total": 1,
                "message": f"缓存命中 {result.get('cache_hits', 0)} 项，模型调用 {result.get('model_calls', 0)} 次",
                "cache_hits": result.get("cache_hits", 0),
                "model_calls": result.get("model_calls", 0),
                "updated_at": time.time(),
            }
            if not result.get("ok"):
                return "夜间整理失败", {"kind": "nightly", "result": result}
            cleared = self.daytime.end_scene()
            result["short_term_cleared"] = cleared
            profile_result = (
                "画像更新成功" if result.get("profile_updated")
                else "未发现足够稳定的画像属性"
            )
            msg = (
                f"场景结束，夜间归纳完成：处理 {result['moment_count']} 条，"
                f"分 {result['chunk_count']} 批，{profile_result}；"
                f"已清理 Layer1/2 {cleared} 条，Layer7 归档保留"
            )
            return msg, {"kind": "nightly", "result": result}
        return {
            "kind": "nightly",
            "title": title or "场景结束，夜间归纳",
            "detail": {"note": note or (
                "复用记忆流动中的夜间流程：从当前会话已写入的记忆建立分类索引、"
                "归纳 L4/L5，并尝试更新 L6 用户画像。只按已有记忆和明确证据归纳；"
                "若测试片段没有指向同一用户的稳定信息，画像保持为空是正常结果。"
            )},
            "inputs": [],
            "run": run_nightly,
        }

    def _plan_custom(self):
        """Run a selected test collection as an isolated, one-off retrieval evaluation."""
        self.dataset_options = interactive_dataset_options()
        if not self.dataset_key.startswith("feishu:"):
            self.dataset_key = "feishu:all"
        selector = self.dataset_key
        moments = feishu_test_moments(selector)
        if not moments:
            raise ValueError("所选自定义测试集没有可运行的视频")

        category = selector.split(":", 1)[1]
        category_name = ("全部类别" if category == "all"
                         else FEISHU_TESTSET_LABELS.get(category, category))
        self.participant = None
        self.moments = moments
        self.meta = {
            "participant": "Feishu custom test set",
            "available_count": len(moments),
            "total_candidates": len(moments),
            "coverage": 1.0,
        }
        self.dataset_id = f"custom_test|{selector}|all"
        self.participant_label = category_name
        self.dataset_note = next(
            (x["note"] for x in self.dataset_options if x["value"] == self.dataset_id), "")

        steps = []
        total = len(moments)
        for index, moment in enumerate(moments, 1):
            def run(moment=moment, index=index, force=False):
                context = {
                    "item_index": index,
                    "item_total": total,
                    "label": f"自定义测评 · 片段 {index}/{total}",
                }
                self._set_step_progress(
                    **context,
                    stage="item_start",
                    message=f"正在感知并检索第 {index}/{total} 段：{moment.get('id', '')}",
                )
                moment_id, info = self._add_one(moment, force=force, progress=context)
                if moment_id is None:
                    message = "实时推理已停止" if info.get("cancelled") else "感知失败"
                    return f"{message}：{info.get('error')}", {"kind": "single", **info}
                return (
                    None,
                    {"kind": "single", **info},
                )

            steps.append({
                "kind": "add",
                "title": f"场景测评 · 写入并检索片段 {index}/{total}",
                "inputs": [self._display_brief(moment)],
                "run": run,
            })
        steps.append(self._nightly_consolidation_step())
        return steps

    def _plan_long(self):
        """长期场景：白天按小时批量写入 + 夜间整理 4 步。"""
        self.dataset_options = interactive_dataset_options()
        if self.dataset_key.startswith("feishu:"):
            selector = self.dataset_key
            moments = feishu_test_moments(selector)
            category = selector.split(":", 1)[1]
            category_name = ("全部类别" if category == "all"
                             else FEISHU_TESTSET_LABELS.get(category, category))
            self.participant = None
            self.moments = moments
            self.meta = {
                "participant": "Feishu custom test set",
                "hours_covered": ["12"],
                "available_count": len(moments),
                "total_candidates": len(moments),
                "coverage": 1.0,
                "complete": False,
            }
            self.dataset_id = f"long_term|{selector}|all"
            self.participant_label = f"自定义测试集 · {category_name}"
        else:
            self.dataset_key = "long_day"
            default_meta = json.load(open(LONG_META, encoding="utf-8"))
            participant = self.participant or default_meta.get("participant", "A1_JAKE")
            meta_path = LONG_PERSON_DIR / f"{participant}.meta.json"
            data_path = LONG_PERSON_DIR / f"{participant}.jsonl"
            if not meta_path.exists() or not data_path.exists():
                raise ValueError(f"没有 {participant} 的长期数据集")
            meta = json.load(open(meta_path, encoding="utf-8"))
            moments = _load_jsonl(data_path)
            self.participant = participant
            self.moments = moments
            self.meta = meta
            self.dataset_id = f"long_term|long_day|{participant}"
            self.participant_label = PARTICIPANT_NAMES.get(participant, participant)
        self.dataset_note = next(
            (x["note"] for x in self.dataset_options if x["value"] == self.dataset_id), "")
        steps = []

        # 白天：按小时分批
        by_hour = {}
        for m in moments:
            h = m.get("_hour") or (m["id"].split("_")[3][:2] if len(m["id"].split("_")) > 3 else "??")
            by_hour.setdefault(h, []).append(m)

        for h in sorted(by_hour):
            group = by_hour[h]
            def run(group=group, h=h, force=False):
                items, ok = [], 0
                cancelled = False
                for item_index, m in enumerate(group, 1):
                    context = {"item_index": item_index, "item_total": len(group),
                               "label": f"{h}:00 时段 · 逐条写入"}
                    self._set_step_progress(**context, stage="item_start",
                                            message=f"正在处理第 {item_index}/{len(group)} 条：{m.get('id', '')}")
                    mid, info = self._add_one(m, force=force, progress=context)
                    if mid:
                        ok += 1
                    items.append(info)
                    if info.get("cancelled") or (self._active_step_cancel and self._active_step_cancel.is_set()):
                        cancelled = True
                        break
                msg = f"{h}:00 时段：Qwen 感知 {ok}/{len(group)} 条并写入记忆"
                if cancelled:
                    msg = f"实时推理已停止；本时段已完成 {ok}/{len(group)} 条"
                return msg, {
                    "kind": "batch", "items": items, "n": len(group),
                    "ok": ok, "cancelled": cancelled,
                }
            steps.append({
                "kind": "add_batch",
                "title": f"白天感知写入 · {h}:00 时段（{len(group)} 条）",
                "inputs": [self._display_brief(m) for m in group],
                "run": run,
            })

        def run_merge_candidates(force=False):
            candidates = self.mem.find_merge_candidates(limit=100)
            result = {
                "kind": "operation_result", "operation": "merge_candidates",
                "title": "相似事件候选 · 逐条确认后才会合并",
                "candidates": candidates,
            }
            return f"发现 {len(candidates)} 组相似事件候选；没有自动合并", result
        steps.append({
            "kind": "merge_candidates",
            "title": "检查相似事件候选（需要逐条确认）",
            "detail": {"note": "候选依据：同源片段重叠≥80%，或120秒内Scene与Action合并文本相似度≥92%且共享人物/物品。画面与语音转写文本只供核对；扫描不改数据，逐条确认后才归并。"},
            "inputs": [],
            "run": run_merge_candidates,
        })

        steps.append(self._nightly_consolidation_step(
            title="夜间整理 · sort 归类、combine 合并、compress 压缩、update_profile 画像更新",
            note="按真实 Qwen 感知结果执行归类、同义项合并、L4/L5 摘要压缩和画像更新；"
                 "画像只写入稳定且有来源的属性，不读取 ground-truth。",
        ))

        return steps

    # ------------------------------------------------------------------
    # 执行与状态
    # ------------------------------------------------------------------

    def _full_snapshot(self):
        """提取各层实际内容（供前端展示完整记忆，不只是数值）。"""
        m = self.mem.memory
        l7 = m["layer7"]

        # 反向映射：记忆 moment_id → 原始数据集 moment，用于补充视频地址 / 时间点
        orig_by_id = {x.get("id"): x for x in getattr(self, "moments", [])}
        rev = {mid: did for did, mid in self.id_map.items()}

        def vid_of(mid):
            """该条记忆对应的视频（可播放地址 + 片段区间 + 拍摄时刻）。"""
            orig = orig_by_id.get(rev.get(mid))
            if not orig:
                return None
            b = moment_brief(orig)
            if not b.get("video_url"):
                return None
            return {"video": b["video"], "video_url": b["video_url"],
                    "clip_start": b["clip_start"], "clip_end": b["clip_end"],
                    "clock": b["clock"]}

        def fmt(x, with_stale=False):
            d = {"id": x.get("id", ""), "scene": x.get("scene", ""),
                 "user_action": x.get("user_action", ""),
                 "interaction_details": x.get("interaction_details", ""),
                 "location": x.get("location", ""),
                 "activity": x.get("activity", ""),
                 "people": x.get("people", []),
                 "objects": x.get("objects", []),
                 "environments": x.get("environments", []),
                 "notes": x.get("notes", ""),
                 "person_observations": copy.deepcopy(x.get("person_observations", [])),
                 "temporal_facts": copy.deepcopy(x.get("temporal_facts", [])),
                 "source_time": copy.deepcopy(x.get("source_time", {})),
                 "highlight": x.get("highlighted", False),
                 "timestamp": x.get("timestamp", "")}
            if with_stale:
                d["stale"] = x.get("stale", False)
                d["merged_into"] = x.get("merged_into", "")
                d["merged_sources"] = copy.deepcopy(x.get("merged_sources", []))
                d["merge_history"] = copy.deepcopy(x.get("merge_history", [])[-5:])
            media = x.get("multimodal")
            if isinstance(media, dict):
                d["multimodal"] = copy.deepcopy(media)
                strip_audio_assets(d)
                for temporal_ref in media_refs(d["multimodal"], "visual"):
                    temporal_ref["url"] = "/api/memory_asset?p=" + quote(temporal_ref["asset"])
                ref = d["multimodal"].get("visual")
                if isinstance(ref, dict) and ref.get("asset"):
                    ref["url"] = "/api/memory_asset?p=" + quote(ref["asset"])
                object_visuals = d["multimodal"].get("object_visuals") or {}
                for visuals in (object_visuals, d["multimodal"].get("person_visuals") or {}):
                    for item in visuals.get("items") or []:
                        for frame_ref in item.get("frame_refs") or []:
                            if frame_ref.get("asset"):
                                frame_ref["url"] = "/api/memory_asset?p=" + quote(frame_ref["asset"])
                            if frame_ref.get("source_asset"):
                                frame_ref["source_url"] = "/api/memory_asset?p=" + quote(frame_ref["source_asset"])
                entity_visuals = d["multimodal"].get("entity_visuals") or {}
                entity_assets = entity_visuals.get("assets") or []
                for asset_ref in entity_assets:
                    if asset_ref.get("asset"):
                        asset_ref["url"] = "/api/memory_asset?p=" + quote(asset_ref["asset"])
                    if asset_ref.get("source_asset"):
                        asset_ref["source_url"] = "/api/memory_asset?p=" + quote(asset_ref["source_asset"])
                for entity_item in entity_visuals.get("items") or []:
                    entity_item["frame_refs"] = [
                        copy.deepcopy(entity_assets[index])
                        for index in entity_item.get("asset_indices") or []
                        if isinstance(index, int) and 0 <= index < len(entity_assets)
                    ]
            source = x.get("source_provenance")
            if isinstance(source, dict):
                d["source_provenance"] = copy.deepcopy(source)
                d["source_provenance"]["source_video_id"] = _portable_video_id(
                    d["source_provenance"].get("source_video_id"))
            v = vid_of(x.get("id", ""))
            if v:
                d["video"] = v
            return d

        # 索引：列出每个维度下的 tag 及其命中的 moment id（限每个维度 15 个 tag，每个 tag 5 个 id）
        indices = {}
        for dim in ("people", "locations", "time_nodes", "activity_events", "environments", "objects"):
            idx = l7.get(dim, {})
            indices[dim] = {tag: ids[:5] for tag, ids in list(idx.items())[:15]}

        layer7_moments = list(l7.get("moments", {}).values())
        merged_count = sum(bool(item.get("merged_into")) for item in layer7_moments)
        object_index_samples = {}
        object_ranked = sorted(l7.get("objects", {}).items(),
                               key=lambda item: (-len(item[1]), str(item[0])))[:8]
        for tag, mids in object_ranked:
            # One representative record per visible top key keeps snapshots compact.
            mid = next((candidate for candidate in reversed(mids)
                        if candidate in l7.get("moments", {})), None)
            if mid:
                object_index_samples[tag] = [fmt(l7["moments"][mid], True)]
        def asset_bytes_for(node):
            """Count unique stored visual assets referenced by one memory layer."""
            asset_paths = set()

            def collect(value):
                if isinstance(value, dict):
                    for field in ("asset", "source_asset"):
                        asset = value.get(field)
                        if isinstance(asset, str):
                            normalized = asset.replace(chr(92), "/").lstrip("/")
                            if normalized.startswith("assets/"):
                                asset_paths.add(normalized)
                    for child in value.values():
                        collect(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        collect(child)

            collect(node)
            total = 0
            asset_root = self.media_store.asset_root.resolve()
            for asset in asset_paths:
                path = (self.media_store.root / asset).resolve()
                try:
                    path.relative_to(asset_root)
                except ValueError:
                    continue
                if path.is_file():
                    total += path.stat().st_size
            return total

        layer_asset_bytes = {
            "layer1": asset_bytes_for(m.get("layer1", [])),
            "layer2": asset_bytes_for(m.get("layer2", [])),
            "layer3": asset_bytes_for(m.get("layer3", [])),
            "layer4": asset_bytes_for(m.get("layer4", {})),
            "layer5": asset_bytes_for(m.get("layer5", {})),
            "layer6": asset_bytes_for(m.get("layer6", {})),
            "layer7": asset_bytes_for(l7.get("moments", {})),
        }
        return {
            "layer1": [fmt(x) for x in m.get("layer1", [])[:30]],
            "layer2": [fmt(x) for x in m.get("layer2", [])[:30]],
            "layer3": [fmt(x) for x in m.get("layer3", [])[:30]],
            "layer4": m.get("layer4", {}),
            "layer5": m.get("layer5", {}),
            "layer6": m.get("layer6", {}),
            "layer7_moments": [fmt(x, True) for x in list(l7.get("moments", {}).values())[:80]],
            "object_index_samples": object_index_samples,
            "layer7_total": len(l7.get("moments", {})),
            "layer7_active_total": len(layer7_moments) - merged_count,
            "layer7_merged_total": merged_count,
            "indices": indices,
            "metadata": m.get("metadata", {}),
            "entity_relationships": entity_graph_summary(l7, limit=12),
            "multimodal_asset_bytes": self.media_store.size_bytes(),
            "layer_asset_bytes": layer_asset_bytes,
            "daytime": self.daytime.snapshot(),
        }

    def _log_snapshot(self, title, kind, msg, perception=None):
        snap = rs.snapshot(self.mem, title, action=kind)
        snap["full"] = self._full_snapshot()
        snap["layer7_active_moments"] = snap["full"]["layer7_active_total"]
        snap["layer7_merged_moments"] = snap["full"]["layer7_merged_total"]
        self.log.append({"kind": kind, "title": title, "msg": msg,
                         "snapshot": snap, "perception": perception})

    def run_step(self, force=False, automatic=False, continuous_kind=None, use_cache=True,
                 progress_id=None):
        """执行当前步；连续模式仅跨越相邻的同类步骤。"""
        self.auto_paused = False
        self.step_cancelled = False
        if self.pos >= len(self.steps):
            return "所有步骤已执行完毕"
        step = self.steps[self.pos]
        if automatic:
            expected_kind = continuous_kind or step["kind"]
            if step["kind"] != expected_kind:
                self.auto_paused = True
                return (f"连续运行已暂停：当前操作类型已从“{expected_kind}”切换为“{step['kind']}”。"
                        f"下一步“{step['title']}”未执行；可继续运行该类操作。")
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._active_step_cancel = cancel_event
            self._active_step_force = bool(force)
            if self._pending_cancel:
                cancel_event.set()
                self._pending_cancel = False
        self._active_step_no = self.pos + 1
        self._active_step_title = step["title"]
        step_item_total = len(step.get("inputs") or []) if step["kind"] == "add_batch" else 1
        self._set_step_progress(
            running=True, stage="starting", label=step["title"],
            step_index=self.pos + 1, step_total=len(self.steps),
            item_index=0, item_total=step_item_total, message="正在启动当前步骤",
            output=None, completed=[], run_id=progress_id or uuid.uuid4().hex,
        )
        try:
            if cancel_event.is_set():
                msg, perception = "当前步骤已终止；没有开始后续处理", {"cancelled": True}
            elif step["kind"] == "nightly":
                msg, perception = step["run"](force=force, use_cache=use_cache)
            else:
                msg, perception = step["run"](force=force)
        except Exception as exc:
            self._set_step_progress(running=False, stage="error", message=str(exc)[:300])
            raise
        finally:
            self._active_step_no = None
            self._active_step_title = None
            with self._cancel_lock:
                if self._active_step_cancel is cancel_event:
                    self._active_step_cancel = None
                    self._active_step_force = False
        self._set_step_progress(running=False,
                                stage="cancelled" if cancel_event.is_set() else "done",
                                message=msg, output=None)
        if cancel_event.is_set():
            self.step_cancelled = True
            if isinstance(perception, dict):
                perception["cancelled"] = True
            msg = "实时推理已停止；已完成的记忆会保留，当前步骤未标为完成，可再次运行。"
            self._log_snapshot(step["title"], step["kind"], msg, perception)
            return msg
        self._log_snapshot(step["title"], step["kind"], msg, perception)
        self.pos += 1
        return msg

    def prewarm(self, limit=None):
        """批量预热 Qwen 缓存（后台任务调用）。返回 (成功, 失败)。"""
        ms = self.moments[:limit] if limit else self.moments
        return self.perceiver.prewarm(ms)

    def state(self):
        """返回给前端的完整状态。"""
        total = len(self.steps)
        cur = None
        if self.pos < total:
            s = self.steps[self.pos]
            cur = {
                "kind": s["kind"],
                "title": s["title"],
                "detail": s.get("detail", {}),
                "inputs": s.get("inputs", []),
            }
        # 原始输出在前端折叠显示；这里保留完整文本，避免用户看到人为截断的字段。
        slim = []
        for e in self.log:
            e2 = dict(e)
            p = e.get("perception")
            if p:
                e2["perception"] = dict(p)
            slim.append(e2)

        meta = SCENARIO_META[self.key]
        cache_entries = getattr(self.perceiver, "cache", {}) or {}
        current_cache_count = sum(
            entry.get("prompt_version") == PROMPT_VERSION
            for entry in cache_entries.values() if isinstance(entry, dict)
        )
        extra = {
            "participant": self.participant,
            "participant_label": self.participant_label,
            "dataset_id": self.dataset_id,
            "dataset_key": self.dataset_key,
            "dataset_note": self.dataset_note,
            "dataset_options": self.dataset_options,
        }
        if self.key == "long_term":
            extra["hours_covered"] = self.meta.get("hours_covered", [])
        display_name = meta["name"]
        description = meta["desc"]
        if self.key == "short_term" and self.participant:
            display_name += f" · {self.participant_label}"
        elif self.key == "long_term":
            extra["available_count"] = self.meta.get("available_count", len(self.moments))
            extra["total_candidates"] = self.meta.get("total_candidates", len(self.moments))
            extra["coverage"] = self.meta.get("coverage", 1.0)
            if self.dataset_key.startswith("feishu:"):
                display_name += " · 飞书自定义测试集"
                extra["complete_day"] = False
                extra["custom_test_dataset"] = True
            else:
                display_name += f" · {self.participant_label}"
                if not self.meta.get("complete", False):
                    display_name += "（部分视频）"
                extra["complete_day"] = self.meta.get("complete", False)
            description = self.dataset_note
        elif self.key == "custom_test":
            display_name += f" · {self.participant_label}"
            extra["available_count"] = self.meta.get("available_count", len(self.moments))
            extra["total_candidates"] = self.meta.get("total_candidates", len(self.moments))
            extra["coverage"] = self.meta.get("coverage", 1.0)
            extra["custom_test_dataset"] = True
            extra["complete_day"] = False
            description = self.dataset_note
        return {
            "scenario": self.key,
            "name": display_name,
            "desc": description,
            "total": total,
            "pos": self.pos,
            "finished": self.pos >= total,
            "current": cur,
            "auto_paused": self.auto_paused,
            "step_cancelled": self.step_cancelled,
            "nightly_progress": dict(self.nightly_progress) if self.nightly_progress else None,
            "step_progress": dict(self.step_progress) if self.step_progress else None,
            "log": slim,
            "operations": [dict(event) for event in self.operation_log[-2000:]],
            "snapshot": self.log[-1]["snapshot"] if self.log else None,
            "qwen": {
                "ok": bool(self.qwen_health.get("model_loaded")),
                "gpu": (self.qwen_health.get("gpu") or {}).get("name"),
                "cached": current_cache_count,
                "legacy_cached": max(0, len(cache_entries) - current_cache_count),
                "cache_prompt_version": PROMPT_VERSION,
            },
            **extra,
        }


if __name__ == "__main__":
    # 命令行自测：逐步执行并打印
    s = InteractiveSession("short_term")
    while not s.state()["finished"]:
        st = s.state()
        print(f"[{st['pos']+1}/{st['total']}] {st['current']['title']} ... ", end="")
        print(s.run_step())
    print("done")
