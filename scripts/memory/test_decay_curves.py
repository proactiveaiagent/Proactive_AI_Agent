"""
test_decay_curves.py — 9.16 三类遗忘曲线测试（设计说明书 §5）
============================================================
验证对象：
  - effective_confidence 三类曲线分派：stable（永久/长有效）、decaying（渐变）、deadline（阶跃）
  - 字段路径默认策略 _default_decay_for_path / _assign_default_decay
  - _finalize_attr 的 decay_type/expires_at 语义校正（带 expires_at 强制 deadline）
  - update_profile 落库后：stable 不 stale、decaying 旧数据衰减、deadline 过期失效

用法：
    cd /data/cxr25/zhx/Proactive_AI_Agent/scripts/memory
    /data/cxr25/zhx/Proactive_AI_Agent/zhx/miniforge3/envs/agent/bin/python test_decay_curves.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import memory as M  # noqa: E402
from memory import (  # noqa: E402
    PersonMemory,
    effective_confidence,
    _default_decay_for_path,
    _assign_default_decay,
    _finalize_attr,
    DECAY_STABLE, DECAY_DECAYING, DECAY_DEADLINE,
    DECAY_RATE_PERMANENT, DECAY_RATE_STABLE, DECAY_RATE_DECAYING,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def section(title):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


def T1_stable_permanent():
    print("\n[T1] stable 曲线：硬身份永久不衰减、偏好行为极低衰减")
    # 硬身份（decay_rate=0）：2000 天后仍不衰减
    attr = {"confidence": 0.9, "last_seen": "2020-01-01T00:00:00",
            "decay_type": DECAY_STABLE, "decay_rate": DECAY_RATE_PERMANENT}
    eff = effective_confidence(attr, now="2026-01-01T00:00:00")
    check("永久字段 2000+ 天不衰减", eff == 0.9, f"got {eff}")

    # 偏好行为（decay_rate=0.001）：约 693 天半衰期，一年（365 天）衰减很小
    attr2 = {"confidence": 0.9, "last_seen": "2025-01-01T00:00:00",
             "decay_type": DECAY_STABLE, "decay_rate": DECAY_RATE_STABLE}
    eff2 = effective_confidence(attr2, now="2026-01-01T00:00:00")
    check("长有效字段一年仅轻微衰减", 0.5 < eff2 < 0.9, f"got {eff2:.4f}")
    check("长有效字段一年不 stale（>0.3）", eff2 > 0.3, f"got {eff2:.4f}")


def T2_decaying():
    print("\n[T2] decaying 曲线：经历状态随时间渐变衰减")
    # 0.005/天，约 139 天半衰期。240 天后约衰减到 0.9*0.995^240 ≈ 0.27 → 接近 stale
    attr = {"confidence": 0.9, "last_seen": "2025-01-01T00:00:00",
            "decay_type": DECAY_DECAYING, "decay_rate": DECAY_RATE_DECAYING}
    eff = effective_confidence(attr, now="2025-08-29T00:00:00")  # 约 240 天
    check("渐变字段 240 天明显衰减", eff < 0.3, f"got {eff:.4f}")

    # 近期（30 天）衰减轻微
    eff2 = effective_confidence(attr, now="2025-01-31T00:00:00")  # 30 天
    check("渐变字段 30 天轻微衰减", 0.75 < eff2 < 0.9, f"got {eff2:.4f}")


def T3_deadline():
    print("\n[T3] deadline 曲线：过期前完全有效，过期瞬间失效")
    attr = {"confidence": 0.8, "decay_type": DECAY_DEADLINE,
            "expires_at": "2026-06-01T00:00:00", "last_seen": "2026-01-01T00:00:00"}
    before = effective_confidence(attr, now="2026-05-31T00:00:00")
    after = effective_confidence(attr, now="2026-06-01T00:00:00")
    check("过期前完全有效（不衰减）", before == 0.8, f"got {before}")
    check("过期瞬间失效（eff=0）", after == 0.0, f"got {after}")


def T4_field_path_default():
    print("\n[T4] 字段路径 → 默认衰减策略")
    check("demographics.name → stable 永久",
          _default_decay_for_path(("demographics", "name")) == (DECAY_STABLE, DECAY_RATE_PERMANENT))
    check("demographics.gender → stable 永久",
          _default_decay_for_path(("demographics", "gender")) == (DECAY_STABLE, DECAY_RATE_PERMANENT))
    check("demographics.occupation → decaying",
          _default_decay_for_path(("demographics", "occupation")) == (DECAY_DECAYING, DECAY_RATE_DECAYING))
    check("preferences.hobbies → stable 长有效",
          _default_decay_for_path(("preferences", "hobbies")) == (DECAY_STABLE, DECAY_RATE_STABLE))
    check("frequent_locations → stable 长有效",
          _default_decay_for_path(("frequent_locations",)) == (DECAY_STABLE, DECAY_RATE_STABLE))


def T5_assign_default_decay():
    print("\n[T5] _assign_default_decay 按路径补默认（保留已有标注）")
    profile = {
        "demographics": {
            "name": {"value": "张三", "confidence": 0.9},           # 未标注 → stable 永久
            "occupation": {"value": "学生", "confidence": 0.8},      # 未标注 → decaying
            "age": {"value": "25", "confidence": 0.7,
                    "decay_type": DECAY_DEADLINE, "expires_at": "2026-01-01T00:00:00"},  # 已标注保留
        },
        "preferences": {"hobbies": [{"value": "摄影", "confidence": 0.8}]},  # → stable 长有效
    }
    _assign_default_decay(profile)
    check("name 补 stable 永久",
          profile["demographics"]["name"]["decay_type"] == DECAY_STABLE
          and profile["demographics"]["name"]["decay_rate"] == DECAY_RATE_PERMANENT)
    check("occupation 补 decaying",
          profile["demographics"]["occupation"]["decay_type"] == DECAY_DECAYING)
    check("hobbies 补 stable 长有效",
          profile["preferences"]["hobbies"][0]["decay_type"] == DECAY_STABLE
          and profile["preferences"]["hobbies"][0]["decay_rate"] == DECAY_RATE_STABLE)
    check("已标注的 age 保留 deadline",
          profile["demographics"]["age"]["decay_type"] == DECAY_DEADLINE
          and profile["demographics"]["age"].get("expires_at") == "2026-01-01T00:00:00")


def T6_finalize_semantic_correction():
    print("\n[T6] _finalize_attr 语义校正：带 expires_at 强制 deadline")
    a = _finalize_attr({"value": "coupon", "confidence": 0.9,
                        "expires_at": "2026-06-01T00:00:00"}, "m1", "2026-01-01T00:00:00")
    check("带 expires_at 强制 deadline", a["decay_type"] == DECAY_DEADLINE)
    check("expires_at 保留", a["expires_at"] == "2026-06-01T00:00:00")

    b = _finalize_attr({"value": "Alice", "confidence": 0.9,
                        "decay_type": DECAY_STABLE, "decay_rate": 0.0}, "m1", "2026-01-01T00:00:00")
    check("LLM 标注 stable 保留", b["decay_type"] == DECAY_STABLE and b["decay_rate"] == 0.0)

    c = _finalize_attr({"value": "Alice", "confidence": 0.9}, "m1", "2026-01-01T00:00:00")
    check("未标注不设 decay_type（由字段路径兜底）", "decay_type" not in c)


def T7_update_profile_stale_by_type(tmpdir):
    print("\n[T7] update_profile 落库后按类型标记 stale")
    m = PersonMemory(memory_dir=str(tmpdir / "case_t7"))
    # 用很旧的 timestamp 写入（模拟久远观察）
    old_ts = "2025-01-01T00:00:00"
    # 同一置信度 0.7、同一久远时间：对比三类曲线抗衰减差异
    m.update_profile({
        "demographics": {"name": {"value": "张三", "confidence": 0.7}},        # 永久：不 stale
        "preferences": {"food": [{"value": "spicy", "confidence": 0.7}]},      # 长有效：不 stale
        "behavior_patterns": {
            "with_agents": [{"value": "偏好提醒", "confidence": 0.7,
                             "decay_type": DECAY_DECAYING}]                     # 渐变：久远 → stale
        },
    }, moment_id="m1", timestamp=old_ts)
    p = m.get_profile()
    name = p["demographics"]["name"]
    food = p["preferences"]["food"][0]
    bhv = p["behavior_patterns"]["with_agents"][0]
    check("永久字段 name 不 stale", name.get("stale") is False, f"stale={name.get('stale')}")
    check("长有效字段 food 不 stale", food.get("stale") is False, f"stale={food.get('stale')}")
    check("渐变字段（久远）stale=True", bhv.get("stale") is True,
          f"stale={bhv.get('stale')} eff={bhv.get('effective_confidence')}")
    check("渐变字段带 effective_confidence", "effective_confidence" in bhv)


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="decay_curves_"))
    try:
        T1_stable_permanent()
        T2_decaying()
        T3_deadline()
        T4_field_path_default()
        T5_assign_default_decay()
        T6_finalize_semantic_correction()
        T7_update_profile_stale_by_type(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 50}\n结果：{PASS} 通过 / {FAIL} 失败")
    sys.exit(0 if FAIL == 0 else 1)
