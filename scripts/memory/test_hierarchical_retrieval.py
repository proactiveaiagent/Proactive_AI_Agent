"""Tests for query precedence across memory Layers 1-5."""
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from memory import PersonMemory
from memory_reasoner import MemoryReasoner


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


def test_layer7_archive_is_searched_when_rolling_windows_are_empty():
    with tempfile.TemporaryDirectory(prefix="layer-query-") as folder:
        memory = new_memory(folder)
        mid = memory.add("archive only marker", "older event", [], [])
        memory.memory["layer1"] = []
        memory.memory["layer2"] = []
        memory.memory["layer3"] = []
        memory._save()
        assert memory.query("archive")  # Existing Layer 7 API still searches the archive.
        hits = memory.query_hierarchical("archive", top_k=5)
        assert len(hits) == 1
        assert hits[0]["layer"] == 7
        assert hits[0]["matched_layers"] == [7]


def test_empty_query_and_nonpositive_limit():
    with tempfile.TemporaryDirectory(prefix="layer-query-") as folder:
        memory = new_memory(folder)
        memory.add("scene", "action", [], [])
        assert memory.query_hierarchical("", top_k=5) == []
        assert memory.query_hierarchical("scene", top_k=0) == []


def test_merged_source_text_remains_searchable_from_canonical_event():
    with tempfile.TemporaryDirectory(prefix="layer-query-merged-") as folder:
        memory = new_memory(folder)
        target_id = memory.add("generic family gathering", "people sit together", [], [],
                               location="home", activity="gathering")
        source_id = memory.add("distinctive glacier parcel marker", "rare source-only event", [], [],
                               location="porch", activity="delivery")
        outcome = memory.merge_moments(source_id, target_id, confirmed_duplicate=True)
        assert outcome["status"] == "merged"
        hits = memory.query_ranked("glacier parcel", top_k=5)
        assert hits
        assert hits[0]["moment"]["id"] == target_id


def test_layer7_bm25_index_is_reused_and_invalidated_after_write():
    with tempfile.TemporaryDirectory(prefix="layer-query-index-") as folder:
        memory = new_memory(folder)
        memory.add("distinctive archive marker", "first action", [], [],
                   location="office", activity="meeting")
        first = memory.query_ranked("distinctive archive", top_k=5)
        cached_index = memory._lexical_search_cache
        second = memory.query_ranked("archive marker", top_k=5)
        assert first and second
        assert memory._lexical_search_cache is cached_index

        new_id = memory.add("newly inserted glacier marker", "new action", [], [],
                            location="office", activity="meeting")
        refreshed = memory.query_ranked("glacier marker", top_k=5)
        assert memory._lexical_search_cache is not cached_index
        assert refreshed and refreshed[0]["moment"]["id"] == new_id


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
        test_layer7_archive_is_searched_when_rolling_windows_are_empty,
        test_empty_query_and_nonpositive_limit,
        test_stale_records_are_excluded_from_both_queries,
        test_layer7_bm25_index_is_reused_and_invalidated_after_write,
        test_merged_source_text_remains_searchable_from_canonical_event,
    ]
    for test in tests:
        test()
        print(f"✅ {test.__name__}")
    print(f"结果：{len(tests)} 组通过 / 0 失败")


def test_retrieval_only_uses_local_plan_and_reports_stage_timings():
    with tempfile.TemporaryDirectory(prefix="retrieval-only-timing-") as folder:
        memory = new_memory(folder)
        mid = memory.add(
            "A person at home places a red cup on the kitchen table.",
            "The camera wearer places a red cup on the table.", [], [],
            location="home", activity="placing a cup", environments=["kitchen"],
        )
        reasoner = MemoryReasoner(memory)
        def unexpected_model_call(*args, **kwargs):
            raise AssertionError("retrieval_only must not invoke the model")
        reasoner._infer = unexpected_model_call
        result = reasoner.reason("", context={
            "source_moment_id": mid,
            "retrieval_only": True,
            "memory_modalities": "text",
        })
        assert result["mode"] == "memory_retrieval"
        assert result["scene_plan_method"] == "structured_source_fields"
        assert result["model_calls"] == 0
        assert set(result["stage_timings_ms"]) == {
            "scene_plan", "memory_recall", "multimodal_linking", "summary_search"
        }
        assert all(value >= 0 for value in result["stage_timings_ms"].values())
