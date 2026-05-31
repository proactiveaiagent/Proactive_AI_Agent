"""
memory3.py — Hierarchical 7-Layer Memory System
================================================
Layer 1 : Current moment  — Part1 scene/action + Part2 needs + Part3 solutions + Part4 feedback
Layer 2 : Same-env moments — same external scene from earlier in the session
Layer 3 : All moments today
Layer 4 : Compressed recent — user's current tasks / periodic life trajectory (text summary)
Layer 5 : Compressed older  — key moments & events that shaped the user / long-term stable patterns (text)
Layer 6 : Compressed older  — user profile & personal preferences / basic background info (text)
Layer 7 : Classified archive — tagged by: time-node, activity-event, person, location

DB Operations
  add       — store Part1-4 data when new data arrives
  update    — Part4 corrects Part1-3 data; habits changed → update layers 4-5-6
  query     — Part2 queries relevant data for need analysis
  retrieve  — pull related data for Part2 analysis; pull Layer-7 data for layers 4-5-6
  compress  — summarise raw layers 4-5-6-7 into text + few images  (triggered after need analysis)
  sort      — classify same-type data under shared tags in layer 7  (triggered at compress time)
  combine   — merge new data with similar existing data in layers 4-5-6-7
  delete    — one-off / accidental / erroneous entries; post-compress duplicates
  highlight — user-confirmed correct data; data that recurs and is merged/similar
"""

import json
import time
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().isoformat()


def _today_str() -> str:
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

def _empty_moment() -> Dict:
    """A single captured moment (goes into Layer 1 → 2 → 3)."""
    return {
        "timestamp": _now_iso(),
        # Part 1
        "scene": "",          # external environment description
        "user_action": "",    # user's own behaviour / action
        # Part 2
        "needs": [],          # [{"need": str, "confidence": float}]
        # Part 3
        "solutions": [],      # [{"need": str, "solution": str}]
        # Part 4 feedback (filled in later, does NOT block Part 1-3)
        "feedback": {
            "confirmed": False,
            "corrections": {},   # {field: corrected_value}
            "user_rating": None
        },
        "highlighted": False,
        "layer": 1
    }


def _empty_db() -> Dict:
    return {
        # --- raw episodic layers ---
        "layer1": [],     # current moment (last N moments, usually just 1)
        "layer2": [],     # same-env moments from this session
        "layer3": [],     # all moments from today

        # --- compressed summary layers (text + optional image paths) ---
        "layer4": {
            "summary": "",
            "current_tasks": [],
            "life_trajectory": "",
            "last_updated": None,
            "source_moment_count": 0
        },
        "layer5": {
            "summary": "",
            "key_events": [],
            "long_term_patterns": "",
            "last_updated": None,
            "source_moment_count": 0
        },
        "layer6": {
            "summary": "",
            "profile": {
                "name": "",
                "basic_info": {},
                "preferences": {},
                "habits": {}
            },
            "last_updated": None,
            "source_moment_count": 0
        },

        # --- classified archive ---
        "layer7": {
            "time_nodes": {},      # {"2024-02-10": [moment_ids]}
            "activity_events": {}, # {"chinese_new_year": [moment_ids]}
            "people": {},          # {"姥姥": [moment_ids]}
            "locations": {},       # {"family_courtyard": [moment_ids]}
            "moments": {}          # {moment_id: moment_dict}  — master store
        },

        # --- housekeeping ---
        "metadata": {
            "total_moments": 0,
            "total_encounters": 0,   # kept for backward compat
            "last_consolidation": None,
            "session_start": _now_iso(),
            "today": _today_str()
        }
    }


# ---------------------------------------------------------------------------
# PersonMemory (drop-in replacement for memory2.PersonMemory)
# ---------------------------------------------------------------------------

class PersonMemory:
    """
    7-layer hierarchical memory with 9 DB operations.

    The key design change vs memory2:
      * Consolidation (compress / sort / combine) is NEVER called inside
        add() or update() — it must be triggered explicitly by the pipeline
        AFTER Part2+3 need-analysis finishes, so it never blocks Part1-3.
      * Part4 feedback is stored via update_feedback() and runs
        asynchronously / after the main pipeline returns.
    """

    MAX_LAYER1 = 5    # keep last N moments in the "current" layer
    MAX_LAYER2 = 20   # same-env history within a session
    MAX_LAYER3 = 100  # all moments today (before daily compress)

    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.memory_file = self.memory_dir / "memory.json"
        self.memory: Dict = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict:
        if self.memory_file.exists():
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Migrate old format if needed
            if "layer1" not in data:
                data = self._migrate_from_v2(data)
            return data
        return _empty_db()

    def _save(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    def _migrate_from_v2(self, old: Dict) -> Dict:
        """Best-effort migration from memory2 format."""
        db = _empty_db()
        # port people → layer7 people index
        for name, pdata in old.get("people", {}).items():
            db["layer7"]["people"][name] = []
        # port locations → layer7 locations index
        for loc in old.get("locations", {}).keys():
            db["layer7"]["locations"][loc] = []
        db["metadata"]["total_encounters"] = old.get("metadata", {}).get("total_encounters", 0)
        return db

    # ------------------------------------------------------------------
    # Internal ID generation
    # ------------------------------------------------------------------

    def _new_moment_id(self) -> str:
        return f"m_{int(time.time() * 1000)}_{self.memory['metadata']['total_moments']}"

    # ------------------------------------------------------------------
    # Operation 1 : ADD
    # ------------------------------------------------------------------

    def add(self,
            scene: str,
            user_action: str,
            needs: List[Dict],
            solutions: List[Dict],
            people: List[str] = None,
            location: str = None,
            activity: str = None,
            extra_notes: str = "") -> str:
        """
        Store a complete Part1+2+3 result.
        Returns the moment_id so Part4 can reference it later.

        Layers updated: 1, 2, 3, 7 (index only)
        Layers 4-6 are updated only during consolidation (not here).
        """
        moment = _empty_moment()
        moment["scene"] = scene
        moment["user_action"] = user_action
        moment["needs"] = needs
        moment["solutions"] = solutions
        if extra_notes:
            moment["notes"] = extra_notes

        moment_id = self._new_moment_id()
        moment["id"] = moment_id

        # ---- Layer 1 (current window) ----
        self.memory["layer1"].append(moment)
        if len(self.memory["layer1"]) > self.MAX_LAYER1:
            # oldest moment graduates to layer2
            old = self.memory["layer1"].pop(0)
            self._graduate_to_layer2(old)

        # ---- Layer 3 (today) ----
        self.memory["layer3"].append(moment)
        if len(self.memory["layer3"]) > self.MAX_LAYER3:
            self.memory["layer3"].pop(0)

        # ---- Layer 7 master store ----
        self.memory["layer7"]["moments"][moment_id] = moment

        # ---- Layer 7 indices ----
        today = _today_str()
        self.memory["layer7"]["time_nodes"].setdefault(today, []).append(moment_id)

        if people:
            for p in people:
                self.memory["layer7"]["people"].setdefault(p, []).append(moment_id)

        if location:
            self.memory["layer7"]["locations"].setdefault(location, []).append(moment_id)

        if activity:
            self.memory["layer7"]["activity_events"].setdefault(activity, []).append(moment_id)

        # ---- metadata ----
        self.memory["metadata"]["total_moments"] += 1
        self.memory["metadata"]["total_encounters"] += 1

        self._save()
        return moment_id

    def _graduate_to_layer2(self, moment: Dict):
        """Move a moment from layer1 to layer2 if env matches current scene."""
        moment["layer"] = 2
        self.memory["layer2"].append(moment)
        if len(self.memory["layer2"]) > self.MAX_LAYER2:
            self.memory["layer2"].pop(0)

    # ------------------------------------------------------------------
    # Operation 2 : UPDATE
    # ------------------------------------------------------------------

    def update_feedback(self, moment_id: str, corrections: Dict = None,
                        confirmed: bool = False, user_rating: int = None):
        """
        Part4 feedback: correct Part1-3 data.
        This is called AFTER the main pipeline and does not block it.
        """
        moment = self.memory["layer7"]["moments"].get(moment_id)
        if not moment:
            return

        fb = moment["feedback"]
        fb["confirmed"] = confirmed
        if corrections:
            fb["corrections"].update(corrections)
            # Apply corrections directly to moment fields
            for field, val in corrections.items():
                if field in moment:
                    moment[field] = val
        if user_rating is not None:
            fb["user_rating"] = user_rating

        # Sync back into layer1/2/3 if still present
        for layer_key in ["layer1", "layer2", "layer3"]:
            for i, m in enumerate(self.memory[layer_key]):
                if m.get("id") == moment_id:
                    self.memory[layer_key][i] = moment

        self._save()

    def update_habits(self, profile_updates: Dict):
        """Update Layer 6 profile when habits / preferences change."""
        profile = self.memory["layer6"]["profile"]
        for key, val in profile_updates.items():
            if isinstance(val, dict) and isinstance(profile.get(key), dict):
                profile[key].update(val)
            else:
                profile[key] = val
        self.memory["layer6"]["last_updated"] = _now_iso()
        self._save()

    # ------------------------------------------------------------------
    # Operation 3 : QUERY  (called by Part2 before need analysis)
    # ------------------------------------------------------------------

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Simple keyword-based retrieval for Part2 need analysis.
        Returns a list of relevant moment dicts from layers 2, 3, 7.
        (A real implementation could use embeddings here.)
        """
        keywords = set(query_text.lower().split())
        scored = []

        candidate_ids = set()
        # Search layer7 indices
        for word in keywords:
            for person in self.memory["layer7"]["people"]:
                if word in person.lower():
                    candidate_ids.update(self.memory["layer7"]["people"][person])
            for loc in self.memory["layer7"]["locations"]:
                if word in loc.lower():
                    candidate_ids.update(self.memory["layer7"]["locations"][loc])
            for evt in self.memory["layer7"]["activity_events"]:
                if word in evt.lower():
                    candidate_ids.update(self.memory["layer7"]["activity_events"][evt])

        for mid in candidate_ids:
            m = self.memory["layer7"]["moments"].get(mid, {})
            text = json.dumps(m, ensure_ascii=False).lower()
            score = sum(1 for kw in keywords if kw in text)
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    # ------------------------------------------------------------------
    # Operation 4 : RETRIEVE  (called by Part2 / consolidation)
    # ------------------------------------------------------------------

    def retrieve(self, people: List[str] = None, location: str = None,
                 activity: str = None, layer: int = None) -> str:
        """
        Pull structured context for Part2 analysis or for building
        layers 4-5-6 during consolidation.
        Returns a human-readable string.
        """
        parts = []

        # Layer 4-6 compressed summaries
        for lnum, lkey in [(4, "layer4"), (5, "layer5"), (6, "layer6")]:
            if layer and layer != lnum:
                continue
            ldata = self.memory[lkey]
            if ldata.get("summary"):
                parts.append(f"[Layer {lnum}] {ldata['summary']}")

        # Layer 7 indexed lookup
        moment_ids = set()
        if people:
            for p in people:
                for key in self.memory["layer7"]["people"]:
                    if p.lower() in key.lower():
                        moment_ids.update(self.memory["layer7"]["people"][key])
        if location:
            for key in self.memory["layer7"]["locations"]:
                if location.lower() in key.lower():
                    moment_ids.update(self.memory["layer7"]["locations"][key])
        if activity:
            for key in self.memory["layer7"]["activity_events"]:
                if activity.lower() in key.lower():
                    moment_ids.update(self.memory["layer7"]["activity_events"][key])

        for mid in list(moment_ids)[:10]:
            m = self.memory["layer7"]["moments"].get(mid)
            if m:
                needs_str = "; ".join(n.get("need", "") for n in m.get("needs", []))
                parts.append(
                    f"[Layer7/{mid}] {m.get('timestamp','')[:10]} | "
                    f"Scene: {m.get('scene','')[:80]} | "
                    f"Needs: {needs_str[:80]}"
                )

        return "\n".join(parts) if parts else "No relevant memory found."

    # ------------------------------------------------------------------
    # Operation 5 : COMPRESS  (called AFTER Part2+3, never during)
    # ------------------------------------------------------------------

    def compress(self, llm_summary: Dict):
        """
        Update layers 4, 5, 6 with LLM-generated compressed summaries.

        llm_summary format:
        {
          "layer4": {"summary": "...", "current_tasks": [...], "life_trajectory": "..."},
          "layer5": {"summary": "...", "key_events": [...], "long_term_patterns": "..."},
          "layer6": {"summary": "...", "profile": {...}}
        }
        """
        for lkey in ["layer4", "layer5", "layer6"]:
            if lkey in llm_summary:
                updates = llm_summary[lkey]
                layer = self.memory[lkey]
                for k, v in updates.items():
                    if isinstance(v, dict) and isinstance(layer.get(k), dict):
                        layer[k].update(v)
                    elif isinstance(v, list) and isinstance(layer.get(k), list):
                        # Merge lists (avoid duplicates)
                        existing = set(str(x) for x in layer[k])
                        for item in v:
                            if str(item) not in existing:
                                layer[k].append(item)
                                existing.add(str(item))
                    else:
                        layer[k] = v
                layer["last_updated"] = _now_iso()
                layer["source_moment_count"] = self.memory["metadata"]["total_moments"]

        self.memory["metadata"]["last_consolidation"] = _now_iso()
        self._save()

    # ------------------------------------------------------------------
    # Operation 6 : SORT  (called together with compress)
    # ------------------------------------------------------------------

    def sort(self, sort_analysis: Dict):
        """
        Re-classify layer7 indices based on LLM analysis.

        sort_analysis format:
        {
          "people": {"姥姥": ["m_xxx", ...]},
          "locations": {"family_courtyard": ["m_xxx", ...]},
          "activity_events": {"chinese_new_year": ["m_xxx", ...]},
          "time_nodes": {"2024-02-10": ["m_xxx", ...]}
        }
        """
        for index_key in ["people", "locations", "activity_events", "time_nodes"]:
            if index_key in sort_analysis:
                for tag, ids in sort_analysis[index_key].items():
                    existing = set(self.memory["layer7"][index_key].get(tag, []))
                    existing.update(ids)
                    self.memory["layer7"][index_key][tag] = list(existing)
        self._save()

    # ------------------------------------------------------------------
    # Operation 7 : COMBINE  (called together with compress)
    # ------------------------------------------------------------------

    def combine(self, canonical_map: Dict):
        """
        Merge near-duplicate layer7 index keys into a canonical form.

        canonical_map format:
        {
          "locations": {"Old fuzzy name": "canonical_name"},
          "activity_events": {"old tag": "canonical tag"}
        }
        """
        for index_key, mapping in canonical_map.items():
            if index_key not in self.memory["layer7"]:
                continue
            index = self.memory["layer7"][index_key]
            for old_key, canonical in mapping.items():
                if old_key in index and old_key != canonical:
                    ids = index.pop(old_key)
                    existing = set(index.get(canonical, []))
                    existing.update(ids)
                    index[canonical] = list(existing)
                    # Update moment references too
                    for mid in existing:
                        m = self.memory["layer7"]["moments"].get(mid)
                        if m:
                            if index_key == "locations" and m.get("location") == old_key:
                                m["location"] = canonical
        self._save()

    # ------------------------------------------------------------------
    # Operation 8 : DELETE
    # ------------------------------------------------------------------

    def delete(self, moment_id: str = None, reason: str = "manual"):
        """
        Remove a one-off / accidental / post-compress duplicate moment.
        """
        if moment_id and moment_id in self.memory["layer7"]["moments"]:
            del self.memory["layer7"]["moments"][moment_id]
            # Remove from indices
            for index_key in ["people", "locations", "activity_events", "time_nodes"]:
                for tag, ids in self.memory["layer7"][index_key].items():
                    if moment_id in ids:
                        ids.remove(moment_id)
            # Remove from raw layers
            for lkey in ["layer1", "layer2", "layer3"]:
                self.memory[lkey] = [m for m in self.memory[lkey]
                                     if m.get("id") != moment_id]
            self._save()
            print(f"🗑️  Deleted moment {moment_id} ({reason})")

    # ------------------------------------------------------------------
    # Operation 9 : HIGHLIGHT
    # ------------------------------------------------------------------

    def highlight(self, moment_id: str):
        """
        Mark a moment as confirmed-correct / high-value.
        Highlighted moments survive delete sweeps and get higher retrieval priority.
        """
        if moment_id in self.memory["layer7"]["moments"]:
            self.memory["layer7"]["moments"][moment_id]["highlighted"] = True
            self._save()

    # ------------------------------------------------------------------
    # Context helpers (called by pipeline)
    # ------------------------------------------------------------------

    def get_context_for_analysis(self, people: List[str] = None,
                                  location: str = None,
                                  activity: str = None) -> str:
        """
        Build a compact context string for the LLM (Part1-3 prompt).
        Queries layers 4→6→7 in order so the most stable/compressed info
        comes first, then recent episodic details.
        """
        lines = ["=== MEMORY CONTEXT ==="]

        # Layer 6 — profile
        profile = self.memory["layer6"]["profile"]
        if profile.get("name") or profile.get("basic_info"):
            lines.append(f"[Profile] {json.dumps(profile, ensure_ascii=False)[:300]}")

        # Layer 5 — long-term patterns
        if self.memory["layer5"]["summary"]:
            lines.append(f"[Long-term] {self.memory['layer5']['summary'][:300]}")

        # Layer 4 — current tasks
        if self.memory["layer4"]["summary"]:
            lines.append(f"[Recent] {self.memory['layer4']['summary'][:300]}")

        # Layer 2 — same-env history
        same_env = self.memory["layer2"][-3:] if self.memory["layer2"] else []
        for m in same_env:
            lines.append(
                f"[SameEnv/{m.get('timestamp','')[:16]}] "
                f"Scene: {m.get('scene','')[:80]} | "
                f"Needs: {'; '.join(n.get('need','') for n in m.get('needs',[]))[:80]}"
            )

        # Layer 7 — person / location / activity
        extra = self.retrieve(people=people, location=location, activity=activity)
        if extra and extra != "No relevant memory found.":
            lines.append(extra)

        return "\n".join(lines) if len(lines) > 1 else "No previous memory found."

    def get_all_memory(self) -> str:
        """Human-readable full memory dump for debugging."""
        lines = ["=== FULL MEMORY DUMP ==="]
        lines.append(f"Total moments: {self.memory['metadata']['total_moments']}")
        lines.append(f"Layer1 (current): {len(self.memory['layer1'])} moments")
        lines.append(f"Layer2 (same-env): {len(self.memory['layer2'])} moments")
        lines.append(f"Layer3 (today): {len(self.memory['layer3'])} moments")

        for lnum, lkey in [(4, "layer4"), (5, "layer5"), (6, "layer6")]:
            s = self.memory[lkey].get("summary", "")
            lines.append(f"Layer{lnum}: {s[:120] if s else '(empty)'}")

        l7 = self.memory["layer7"]
        lines.append(f"Layer7 people: {list(l7['people'].keys())[:10]}")
        lines.append(f"Layer7 locations: {list(l7['locations'].keys())[:10]}")
        lines.append(f"Layer7 events: {list(l7['activity_events'].keys())[:10]}")
        lines.append(f"Layer7 moments stored: {len(l7['moments'])}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Backward-compatible shim (so old code calling memory.update() works)
    # ------------------------------------------------------------------

    def update(self, people: List[str], location: str, notes: Optional[str] = None):
        """Backward-compatible shim — stores a minimal moment."""
        self.add(
            scene=location,
            user_action="",
            needs=[],
            solutions=[],
            people=people,
            location=location,
            extra_notes=notes or ""
        )

    def get_context(self, people: List[str] = None, location: str = None) -> str:
        """Backward-compatible shim."""
        return self.get_context_for_analysis(people=people, location=location)


# ---------------------------------------------------------------------------
# HintMemory — extra memory bank for user-supplied "when X seen → output need Y"
# ---------------------------------------------------------------------------
#
# This is a SEPARATE memory store from PersonMemory. It is *not* episodic;
# it holds user-curated rules that get injected into the analysis prompt so
# the agent can recognise needs the base model would otherwise miss.
#
# Each hint is a small dict:
#   {
#     "id":       "h_<timestamp>",
#     "when":     "trigger description (objects/scene/conditions to look for)",
#     "then":     "the specific need / action the agent should output when the trigger matches",
#     "added_at": "ISO timestamp"
#   }
#
# Public ops:
#   add(when, then)              -> hint_id
#   remove(hint_id)              -> bool
#   list()                       -> List[Dict]
#   clear()                      -> None
#   format_for_prompt()          -> str  (block to splice into the LLM prompt)
# ---------------------------------------------------------------------------


class HintMemory:
    """User-curated trigger→need rules, stored alongside PersonMemory."""

    def __init__(self, memory_dir: str = "memory", filename: str = "hints.json"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.hints_file = self.memory_dir / filename
        self.hints: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if self.hints_file.exists():
            try:
                with open(self.hints_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception:
                pass
        return []

    def _save(self):
        with open(self.hints_file, "w", encoding="utf-8") as f:
            json.dump(self.hints, f, indent=2, ensure_ascii=False)

    def add(self, when: str, then: str) -> str:
        hint_id = f"h_{int(time.time() * 1000)}_{len(self.hints)}"
        self.hints.append({
            "id": hint_id,
            "when": when.strip(),
            "then": then.strip(),
            "added_at": _now_iso(),
        })
        self._save()
        return hint_id

    def remove(self, hint_id: str) -> bool:
        before = len(self.hints)
        self.hints = [h for h in self.hints if h.get("id") != hint_id]
        changed = len(self.hints) != before
        if changed:
            self._save()
        return changed

    def list(self) -> List[Dict]:
        return list(self.hints)

    def clear(self):
        self.hints = []
        self._save()

    def format_for_prompt(self) -> str:
        """Return a prompt-ready block, or an empty string if no hints."""
        if not self.hints:
            return ""
        lines = ["=== USER HINT RULES (apply these when the trigger is observed) ==="]
        for i, h in enumerate(self.hints, 1):
            lines.append(f"  {i}. WHEN: {h.get('when','')}")
            lines.append(f"     THEN output need: {h.get('then','')}")
        lines.append("=== END USER HINT RULES ===")
        return "\n".join(lines)
