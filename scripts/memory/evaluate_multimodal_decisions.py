"""Reproducible paired reviews with real, source-aligned EgoLife media.

These cases have official event/transcript annotations, not official next-action
answers. Report evidence integrity separately from qualitative suggestion quality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import PersonMemory
from memory_reasoner import MemoryReasoner

CASES = [
    ("jake_keep_box", "Jake：保留盒子与镜布", "DAY1_A1_JAKE_11120000_en",
     "结合可见整理动作和明确的保留要求，避免把要保留的东西一并丢弃。"),
    ("alice_enclosure", "Alice：装配硬盘盒", "DAY1_A2_ALICE_11120000_en",
     "核对是否观察到装配中的物品；不凭单帧猜测电源、故障或完成状态。"),
    ("lucia_flowers", "Lucia：拆包与整理花束", "DAY1_A4_LUCIA_11360000_en",
     "核对花束、剪切工具和操作状态；没有明确需要时允许不提醒。"),
    ("tasha_meat", "Tasha：切肉困难", "DAY1_A3_TASHA_11543000_en",
     "区分画面中的操作、语音中的困难和一般安全建议；不猜测人物身份。"),
    ("alice_shower", "Jake 视角：Alice 说想洗澡", "DAY1_A1_JAKE_22053000_en",
     "画面是电子设备场景而语音涉及洗澡；不能把未拍到洗澡误判为文字或媒体错配。"),
]
CONTROLLED_GOALS = {
    "packing_goal": "I am about to discard all the packaging and contents from this unboxing session. Based on the latest event and its actual frames, what should I keep or throw away?",
    "enclosure_goal": "I want to connect power to this hard drive enclosure now. Based on the latest event and actual frames, can you confirm assembly is complete, and what should I check next?",
}
CASES += [
    ("packing_goal", "构造当前目标：准备丢弃包装", "DAY1_A1_JAKE_11120000_en",
     "目标为评测构造，媒体和已存保留/丢弃指令是真实事件；应区分盒子、内衬和镜布。"),
    ("enclosure_goal", "构造当前目标：装配后准备接电", "DAY1_A2_ALICE_11120000_en",
     "目标为评测构造，媒体来自真实装配片段；不能凭模糊采样帧断言安全或装配完成。"),
]


def save(path: Path, value: dict):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def integrity(answer: dict, archive: dict, mode: str) -> dict:
    mm = answer.get("multimodal_inference") or {}
    checks = []
    citations = (answer.get("recommendation") or {}).get("evidence_ids") or []
    checks.append({"name": "引用仅来自当时可用的记忆", "passed": all(mid in archive for mid in citations)})
    observations = mm.get("source_observations") or []
    checks.append({"name": "网页展现保留逐帧真实秒数和原始观察", "passed":
                   (answer.get("recommendation") or {}).get("multimodal_observations", []) == observations})
    for row in observations:
        for source in row.get("sources") or []:
            mid, at = source.get("memory_id"), source.get("frame_at_s")
            media = (archive.get(mid) or {}).get("multimodal") or {}
            refs = media.get("visual_frames") or [media.get("visual") or {}]
            checks.append({"name": "视觉观察秒数与送入资产一致", "passed":
                           mid in archive and any(ref.get("at_s") == at for ref in refs)})
    if mode == "text":
        checks.append({"name": "文本消融没有偷偷使用画面或音频", "passed":
                       mm.get("visual_frames_used") == 0 and mm.get("audio_transcripts_used") == 0 and not observations})
    else:
        checks.append({"name": "多模态条件实际送入当前事件画面", "passed": mm.get("visual_frames_used", 0) > 0})
    review = answer.get("evidence_review") or {}
    checks.append({"name": "建议复核原句均匹配已存来源", "passed":
                   review.get("status") == "completed" and review.get("source_quotes_verified") is True})
    return {"checks": checks, "passed": sum(x["passed"] for x in checks), "total": len(checks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/egolife/runs/multimodal_aligned_v8_20261007/memory.json")
    parser.add_argument("--output-dir", default="data/egolife/runs/multimodal_decision_review_v40_20261007")
    args = parser.parse_args()
    source = ROOT / args.source
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads(source.read_text(encoding="utf-8"))
    implementation = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
        "code/memory_reasoner.py", "code/memory_audio.py", "code/multimodal_memory.py", "code/memory.py",
        "scripts/memory/evaluate_multimodal_decisions.py")}
    source_by_id = {m["multimodal"]["source_id"]: m for m in original["layer7"]["moments"].values()}
    path = output / "scenario_review.json"
    report = json.loads(path.read_text()) if path.exists() else {
        "title": "真实时序证据与文本消融复核", "version": "v40",
        "source_memory": str(source.relative_to(ROOT)),
        "source_moments": len(source_by_id),
        "review_type": "paired_qualitative_review_not_action_accuracy_benchmark",
        "annotation_note": "事件和语音标注来自 EgoLife；下一步建议没有官方标准答案。",
        "implementation_sha256": implementation,
        "source_memory_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scenarios": {},
    }
    if report.get("implementation_sha256") != implementation:
        raise ValueError("Evaluation code changed; use a fresh output directory instead of mixing results")
    for case_id, title, source_id, target in CASES:
        active = source_by_id[source_id]
        camera = "_".join(source_id.split("_")[1:3])
        # Same camera/user and past events only; no future text, indices or summaries.
        archive = {mid: copy.deepcopy(m) for mid, m in original["layer7"]["moments"].items()
                   if m.get("timestamp", "") <= active["timestamp"] and
                   "_".join(m["multimodal"]["source_id"].split("_")[1:3]) == camera}
        item = report["scenarios"].setdefault(case_id, {
            "title": title, "situation": CONTROLLED_GOALS.get(case_id) or "无额外问题，主动分析当前事件：" + active["scene"],
            "case_type": "controlled_goal_real_media" if case_id in CONTROLLED_GOALS else "real_event_proactive_scan",
            "source_id": source_id, "review_target": target,
            "available_history_count": len(archive), "cutoff_time": active["timestamp"], "ablations": {},
        })
        for mode in ("all", "text"):
            if mode in item["ablations"]:
                continue
            folder = output / case_id / mode
            folder.mkdir(parents=True, exist_ok=True)
            db = copy.deepcopy(original)
            db["layer7"]["moments"] = archive
            for key, value in db["layer7"].items():
                if key != "moments" and isinstance(value, dict):
                    db["layer7"][key] = {tag: [mid for mid in ids if mid in archive]
                                            for tag, ids in value.items() if isinstance(ids, list)}
            for layer in ("layer1", "layer2", "layer3"):
                db[layer] = [m for m in db.get(layer, []) if m.get("id") in archive]
            db["layer4"] = {}
            db["layer5"] = {}
            db["layer6"] = {"profile": {}}
            save(folder / "memory.json", db)
            if mode == "all" and not (folder / "assets").exists():
                shutil.copytree(source.parent / "assets", folder / "assets")
            memory = PersonMemory(str(folder))
            answer = MemoryReasoner(memory).reason(CONTROLLED_GOALS.get(case_id) or "", {
                "source_moment_id": active["id"], "proactive": case_id not in CONTROLLED_GOALS, "memory_modalities": mode})
            answer["integrity_review"] = integrity(answer, archive, mode)
            save(folder / "result.json", answer)
            item["ablations"][mode] = answer
            if mode == "all":
                item.update({key: answer.get(key) for key in (
                    "recommendation", "model_recommendation", "policy_adjustments", "multimodal_inference",
                    "retrieval_rounds", "candidates", "search_plan", "elapsed_ms", "evidence_review")})
                item["perception_before_retrieval"] = answer.get("perception_before_retrieval") or {}
            report["completed_conditions"] = sum(len(row["ablations"]) for row in report["scenarios"].values())
            report["planned_conditions"] = len(CASES) * 2
            save(path, report)
            print(json.dumps({"case": case_id, "mode": mode,
                              "integrity": answer["integrity_review"],
                              "action": (answer.get("recommendation") or {}).get("action"),
                              "visual": (answer.get("multimodal_inference") or {}).get("visual_frames_used"),
                              "audio": (answer.get("multimodal_inference") or {}).get("audio_transcripts_used")},
                             ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
