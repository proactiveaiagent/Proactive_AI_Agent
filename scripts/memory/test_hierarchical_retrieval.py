"""Tests for query precedence across memory Layers 1-5."""
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import PersonMemory


def new_memory(folder):
    return PersonMemory(memory_dir=folder)


def test_recent_layer_wins():
    with tempfile.TemporaryDirectory(prefix="layer-query-") as folder:
        memory = new_memory(folder)
        memory.add("urgent meeting", "urgent action", [], [], location="office",
                   activity="meeting", environments=["office"])
        memory.add("urgent follow-up", "urgent action", [], [], location="office",
                   activity="meeting", environments=["office"])
        memory.compress({
            "layer4": {"summary": "urgent daily summary"},
            "layer5": {"summary": "urgent long-term pattern"},
        })
        hits = memory.query_hierarchical("urgent", top_k=2)
        assert [hit["layer"] for hit in hits] == [1, 1]
        assert len({hit["moment"]["id"] for hit in hits}) == 2


def test_layer2_precedes_layer3_duplicate():
    with tempfile.TemporaryDirectory(prefix="layer-query-") as folder:
        memory = new_memory(folder)
        for i in range(6):
            scene = "rare marker first event" if i == 0 else f"ordinary event {i}"
            memory.add(scene, scene, [], [], location="office", activity="work",
                       environments=["office"])
        hits = memory.query_hierarchical("rare marker", top_k=1)
        assert len(hits) == 1
        assert hits[0]["layer"] == 2
        assert hits[0]["moment"]["scene"] == "rare marker first event"


def test_layer4_then_layer5_fallback():
    with tempfile.TemporaryDirectory(prefix="layer-query-") as folder:
        memory = new_memory(folder)
        memory.compress({
            "layer4": {"summary": "bioluminescent daily note"},
            "layer5": {"summary": "long-term astronomy habit"},
        })
        hits = memory.query_hierarchical("bioluminescent", top_k=2)
        assert len(hits) == 1 and hits[0]["layer"] == 4
        hits = memory.query_hierarchical("astronomy", top_k=2)
        assert len(hits) == 1 and hits[0]["layer"] == 5


def test_layer7_is_not_in_layer1_to_5_query():
    with tempfile.TemporaryDirectory(prefix="layer-query-") as folder:
        memory = new_memory(folder)
        mid = memory.add("archive only marker", "older event", [], [])
        memory.memory["layer1"] = []
        memory.memory["layer2"] = []
        memory.memory["layer3"] = []
        memory._save()
        assert memory.query("archive")  # Existing Layer 7 API still searches the archive.
        assert memory.query_hierarchical("archive", top_k=5) == []


def test_empty_query_and_nonpositive_limit():
    with tempfile.TemporaryDirectory(prefix="layer-query-") as folder:
        memory = new_memory(folder)
        memory.add("scene", "action", [], [])
        assert memory.query_hierarchical("", top_k=5) == []
        assert memory.query_hierarchical("scene", top_k=0) == []


def test_stale_records_are_excluded_from_both_queries():
    with tempfile.TemporaryDirectory(prefix="layer-query-") as folder:
        memory = new_memory(folder)
        mid = memory.add("distinctive stale marker", "stale action", [], [])
        memory.memory["layer7"]["moments"][mid]["stale"] = True
        for layer_no in (1, 2, 3):
            for moment in memory.memory[f"layer{layer_no}"]:
                if moment.get("id") == mid:
                    moment["stale"] = True
        memory._save()
        assert memory.query("distinctive", top_k=5) == []
        assert memory.query_hierarchical("distinctive", top_k=5) == []


if __name__ == "__main__":
    tests = [
        test_recent_layer_wins,
        test_layer2_precedes_layer3_duplicate,
        test_layer4_then_layer5_fallback,
        test_layer7_is_not_in_layer1_to_5_query,
        test_empty_query_and_nonpositive_limit,
        test_stale_records_are_excluded_from_both_queries,
    ]
    for test in tests:
        test()
        print(f"✅ {test.__name__}")
    print(f"结果：{len(tests)} 组通过 / 0 失败")
