"""
vr_assistant3.py — Redesigned pipeline
========================================
Timing split:
  • PHASE A  (Part 1-3): parallel extraction → transcribe → analyse → memory add
  • PHASE B  (Part 4) : AR display + user feedback  ← separate timer
  • PHASE C  (Post-4) : memory consolidation  ← triggered AFTER Part4 feedback,
                        never during Part1-3, so it never blocks response time

Memory: uses memory3.PersonMemory (7-layer, 9 operations)
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor
from memory import PersonMemory
import time
from functools import wraps
import requests
import json
import re
import threading


# ---------------------------------------------------------------------------
# Timer decorator
# ---------------------------------------------------------------------------

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️  {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


# ---------------------------------------------------------------------------
# VRAssistant
# ---------------------------------------------------------------------------

class VRAssistant:
    def __init__(self, video_path, output_dir="output",
                 num_threads=None, qwen_api_url="http://localhost:8000"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.qwen_api_url = qwen_api_url

        # CPU threading
        if num_threads is None:
            num_threads = int(os.cpu_count() * 0.75)
        torch.set_num_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Device: {self.device}")
        if self.device == "cuda":
            print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
            print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"🔧 CPU cores: {os.cpu_count()}  threads: {num_threads}")

        # Separate timing buckets
        #   phase_a = Part1-3  (extraction → analysis → memory add)
        #   phase_b = Part4    (AR display + user feedback)
        #   phase_c = Post-4   (consolidation)
        self.timings_phase_a: dict = {}
        self.timings_phase_b: dict = {}
        self.timings_phase_c: dict = {}

        # Memory
        print("Initialising memory (7-layer)...")
        self.memory = PersonMemory()

        # API health check
        print("Checking API server (Qwen + Whisper)...")
        try:
            resp = requests.get(f"{self.qwen_api_url}/health", timeout=2)
            h = resp.json()
            ok_q = h.get("model_loaded", False)
            ok_w = h.get("whisper_loaded", False)
            if ok_q and ok_w:
                print("✅ API server ready (Qwen ✓  Whisper ✓)")
            else:
                missing = [m for m, ok in [("Qwen", ok_q), ("Whisper", ok_w)] if not ok]
                print(f"⚠️  API server running but missing: {', '.join(missing)}")
        except Exception:
            print("❌ API server not available. Run: python api_server.py")

    # -----------------------------------------------------------------------
    # ─── PHASE A helpers ─────────────────────────────────────────────────
    # -----------------------------------------------------------------------

    @timer
    def extract_first_frame(self) -> str:
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Failed to extract first frame")
        path = self.output_dir / "first_frame.jpg"
        cv2.imwrite(str(path), frame)
        return str(path)

    @timer
    def extract_audio(self) -> str:
        path = self.output_dir / "audio.wav"
        video = VideoFileClip(self.video_path)
        video.audio.write_audiofile(str(path), verbose=False, logger=None)
        video.close()
        return str(path)

    @timer
    def transcribe_audio(self, audio_path: str) -> dict:
        try:
            with open(audio_path, "rb") as f:
                resp = requests.post(
                    f"{self.qwen_api_url}/transcribe",
                    files={"audio": (Path(audio_path).name, f, "audio/wav")},
                    timeout=120
                )
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("segments") or not data.get("full_text"):
                    data["no_speech"] = True
                    print("⚠️  No speech detected — visual-only analysis.")
                return data
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"⚠️  Transcription failed ({e}). Proceeding with visual-only analysis.")
            return {"language": "unknown", "full_text": "", "segments": [], "no_speech": True}

    def format_transcript(self, td: dict) -> str:
        if td.get("no_speech") or not td.get("segments"):
            return "Language: unknown\n\n[No speech detected — visual-only analysis]\n"
        out = f"Language: {td['language']}\n\nTranscript with timestamps:\n"
        for seg in td["segments"]:
            out += f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}\n"
        return out

    @timer
    def analyze_with_qwen(self, frame_path: str,
                          transcript_text: str, memory_context: str) -> str:
        """
        Part 1 — scene / people / action recognition
        Part 2 — need analysis
        Part 3 — solution generation
        All three are combined in a single LLM call to avoid latency.
        """
        no_speech = "[No speech detected" in transcript_text
        ts_section = (
            "Transcript: [No audio transcript — base analysis on visual information only]"
            if no_speech else f"Transcript:\n{transcript_text}"
        )

        prompt = f"""You are a proactive personal assistant analysing first-person VR glasses footage.

{memory_context}

{ts_section}

Based on the image (first-person POV), the transcript (if any), and memory above, provide:

## PART 1 — Scene Recognition
- Location: [specific location]
- Time/Occasion: [time, festival, event …]
- People: [name1 (relationship), name2 (relationship), …]
- User Action: [what the user is doing]

## PART 2 — Need Analysis
Identify the user's top 3 needs in priority order. The first need MUST address who the people in view are.
For each need:
- Need [N]: [description]  (Confidence: [0-1])

## PART 3 — Solutions
For each need above, provide a concrete actionable solution:
- Solution [N]: [action / information to present to user via AR]

Keep the response structured and concise."""

        with open(frame_path, "rb") as f:
            resp = requests.post(
                f"{self.qwen_api_url}/analyze",
                files={"image": f},
                data={"prompt": prompt}
            )
        if resp.status_code == 200:
            return resp.json()["analysis"]
        raise Exception(f"API error: {resp.json().get('error', 'Unknown')}")

    def parse_analysis(self, text: str) -> dict:
	    """Extract structured fields from the Part1-3 response."""
	    result = {
	        "people": [],
	        "location": None,
	        "user_action": "",
	        "scene": "",
	        "needs": [],
	        "solutions": []
	    }
	    lines = text.split("\n")
	    for line in lines:
	        ll = line.lower().strip()
	        if ll.startswith("- location:") or ll.startswith("location:"):
	            result["location"] = line.split(":", 1)[-1].strip()
	        elif ll.startswith("- people:") or ll.startswith("people:"):
	            raw = line.split(":", 1)[-1].strip()
	            raw_people = re.split(r"[,，、]", raw)
	            result["people"] = [
	                re.sub(r"（[^）]*）|\([^)]*\)", "", p).strip()
	                for p in raw_people
	                if re.sub(r"（[^）]*）|\([^)]*\)", "", p).strip()
	                and re.sub(r"（[^）]*）|\([^)]*\)", "", p).strip().lower()
	                not in ("none", "n/a")
	            ]
	        elif ll.startswith("- user action:") or ll.startswith("user action:"):
	            result["user_action"] = line.split(":", 1)[-1].strip()
	        elif re.match(r"^-?\s*need\s*\[?\d", ll):
	            body = line.split(":", 1)[-1].strip()
	            conf_match = re.search(r"\(confidence:\s*([0-9.]+)\)", body, re.I)
	            conf = float(conf_match.group(1)) if conf_match else 0.8
	            need_text = re.sub(r"\(confidence:[^)]+\)", "", body, flags=re.I).strip()
	            result["needs"].append({"need": need_text, "confidence": conf})
	        elif re.match(r"^-?\s*solution\s*\[?\d", ll):
	            body = line.split(":", 1)[-1].strip()
	            result["solutions"].append({"solution": body})
	
	    result["scene"] = result.get("location", "")
	    for i, sol in enumerate(result["solutions"]):
	        if i < len(result["needs"]):
	            sol["need"] = result["needs"][i]["need"]
	
	    return result

    # -----------------------------------------------------------------------
    # ─── PHASE A — main Part1-3 pipeline ─────────────────────────────────
    # -----------------------------------------------------------------------

    def run_phase_a(self) -> dict:
        """
        Runs the full Part1-3 pipeline.
        Returns everything needed for Phase B (Part4) and Phase C (consolidation).
        Populates self.timings_phase_a.
        """
        phase_start = time.time()

        print("\n" + "=" * 60)
        print("🚀 PHASE A — Part1-3 (extraction → analysis → memory add)")
        print("=" * 60)

        # ── parallel extraction ──────────────────────────────────────────
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as ex:
            ff = ex.submit(self.extract_first_frame)
            af = ex.submit(self.extract_audio)
            frame_path = ff.result()
            audio_path = af.result()
        self.timings_phase_a["parallel_extraction"] = time.time() - t0
        print(f"✅ Extraction: {self.timings_phase_a['parallel_extraction']:.2f}s")

        # ── transcription ────────────────────────────────────────────────
        t0 = time.time()
        transcript_data = self.transcribe_audio(audio_path)
        self.timings_phase_a["transcribe"] = time.time() - t0
        formatted_transcript = self.format_transcript(transcript_data)
        if not transcript_data.get("no_speech"):
            print(f"📝 Language: {transcript_data['language']}")
            print(formatted_transcript)

        # ── memory context retrieval ─────────────────────────────────────
        t0 = time.time()
        memory_context = self.memory.get_all_memory()
        self.timings_phase_a["memory_retrieval"] = time.time() - t0
        print(f"\n{memory_context}\n")

        # ── Part1-3 LLM analysis ─────────────────────────────────────────
        t0 = time.time()
        analysis_text = self.analyze_with_qwen(frame_path, formatted_transcript, memory_context)
        self.timings_phase_a["analyze_qwen"] = time.time() - t0
        print("\n" + "=" * 60)
        print("📊 PART1-3 ANALYSIS")
        print("=" * 60)
        print(analysis_text)

        # ── memory ADD (add, not consolidate) ────────────────────────────
        t0 = time.time()
        parsed = self.parse_analysis(analysis_text)
        moment_id = self.memory.add(
            scene=parsed["scene"],
            user_action=parsed["user_action"],
            needs=parsed["needs"],
            solutions=parsed["solutions"],
            people=parsed["people"],
            location=parsed["location"],
            extra_notes=transcript_data.get("language", "")
        )
        self.timings_phase_a["memory_add"] = time.time() - t0
        print(f"💾 Memory stored: moment_id={moment_id}  "
              f"people={parsed['people']}  location={parsed['location']}")

        self.timings_phase_a["total_phase_a"] = time.time() - phase_start

        return {
            "frame_path": frame_path,
            "audio_path": audio_path,
            "transcript": transcript_data,
            "formatted_transcript": formatted_transcript,
            "analysis_text": analysis_text,
            "parsed": parsed,
            "moment_id": moment_id,
        }

    # -----------------------------------------------------------------------
    # ─── PHASE B — Part4 (AR display + feedback) ─────────────────────────
    # -----------------------------------------------------------------------

    def run_phase_b(self, phase_a_result: dict) -> dict:
        """
        Part4: decide HOW to present Part3 solutions on the AR device,
        then collect user feedback.

        Timing is tracked in self.timings_phase_b separately from Phase A.
        Consolidation is NOT triggered here.
        """
        phase_start = time.time()

        print("\n" + "=" * 60)
        print("🥽 PHASE B — Part4 (AR presentation + user feedback)")
        print("=" * 60)

        parsed = phase_a_result["parsed"]
        moment_id = phase_a_result["moment_id"]

        # ── AR presentation decision ─────────────────────────────────────
        t0 = time.time()
        ar_plan = self._decide_ar_presentation(parsed)
        self.timings_phase_b["ar_decision"] = time.time() - t0

        print("\n📺 AR Presentation Plan:")
        print(json.dumps(ar_plan, indent=2, ensure_ascii=False))

        # ── (Simulated) present on AR device ────────────────────────────
        t0 = time.time()
        self._present_on_ar(ar_plan, parsed)
        self.timings_phase_b["ar_presentation"] = time.time() - t0

        # ── User feedback (in real system this would await hardware input) ─
        t0 = time.time()
        feedback = self._collect_feedback_simulated(parsed, moment_id)
        self.timings_phase_b["feedback_collection"] = time.time() - t0

        if feedback:
            self.memory.update_feedback(
                moment_id=moment_id,
                corrections=feedback.get("corrections"),
                confirmed=feedback.get("confirmed", False),
                user_rating=feedback.get("rating")
            )
            if feedback.get("confirmed"):
                self.memory.highlight(moment_id)
                print(f"⭐ Moment {moment_id} highlighted as confirmed-correct.")

        self.timings_phase_b["total_phase_b"] = time.time() - phase_start

        return {
            "ar_plan": ar_plan,
            "feedback": feedback,
            "moment_id": moment_id,
        }

    def _decide_ar_presentation(self, parsed: dict) -> dict:
        """
        Decide modality, interaction mode, and timing for each solution.

        Rules (from spec image):
          Modality:
            noisy env → text
            user focused (e.g. driving) → voice
            physical world info (e.g. nav) → 3D
            normal → voice + text

          Interaction:
            need confidence < 0.7 → ask for confirmation
            solution needs auth (payment etc.) → require confirmation
            high confidence simple info → direct display

          Timing:
            simple high-confidence → immediate
            urgent service (emergency) → immediate
            auth-required → wait for interaction
        """
        plan = {"solutions": []}
        for i, (need, sol) in enumerate(zip(parsed["needs"], parsed["solutions"])):
            conf = need.get("confidence", 0.8)
            need_text = need.get("need", "").lower()

            # Modality heuristics
            if any(k in need_text for k in ("navigate", "direction", "map", "route")):
                modality = "3D"
            elif conf < 0.6:
                modality = "text"
            else:
                modality = "voice+text"

            # Interaction heuristics
            if conf < 0.7:
                interaction = "need_confirmation"
            elif any(k in need_text for k in ("pay", "purchase", "buy", "order")):
                interaction = "solution_confirmation"
            else:
                interaction = "direct_display"

            # Timing heuristics
            if any(k in need_text for k in ("emergency", "urgent", "danger", "help")):
                timing = "immediate"
            elif interaction in ("need_confirmation", "solution_confirmation"):
                timing = "wait_interaction"
            else:
                timing = "immediate" if conf >= 0.8 else "delayed"

            plan["solutions"].append({
                "index": i + 1,
                "need": need.get("need", ""),
                "solution": sol.get("solution", ""),
                "modality": modality,
                "interaction": interaction,
                "timing": timing,
                "confidence": conf
            })
        return plan

    def _present_on_ar(self, ar_plan: dict, parsed: dict):
        """
        Simulate AR presentation output.
        In production this would call the AR device API.
        """
        print("\n--- AR Device Output (simulated) ---")
        for item in ar_plan["solutions"]:
            icon = {"voice+text": "🔊📝", "text": "📝", "3D": "🌐", "voice": "🔊"}.get(
                item["modality"], "📝"
            )
            timing_icon = "⚡" if item["timing"] == "immediate" else "⏳"
            interact_icon = "✅" if item["interaction"] == "direct_display" else "❓"
            print(f"  {icon}{timing_icon}{interact_icon} [{item['modality']}] "
                  f"Need {item['index']}: {item['solution'][:80]}")
        print("--- End AR Output ---\n")

    def _collect_feedback_simulated(self, parsed: dict, moment_id: str) -> dict:
        """
        Simulated feedback collection.
        In production this would listen for voice/gesture/tap input.
        Returns a feedback dict or None if no feedback.
        """
        # For now, auto-confirm if all needs have confidence >= 0.8
        all_high_conf = all(n.get("confidence", 0) >= 0.8 for n in parsed["needs"])
        if all_high_conf:
            print("✅ Auto-confirmed (all needs confidence ≥ 0.8)")
            return {"confirmed": True, "corrections": {}, "rating": 5}
        else:
            print("⚠️  Low-confidence needs — skipping auto-confirm (awaiting user input).")
            return {"confirmed": False, "corrections": {}, "rating": None}

    # -----------------------------------------------------------------------
    # ─── PHASE C — Post-Part4 consolidation ──────────────────────────────
    # -----------------------------------------------------------------------

    def run_phase_c(self, phase_b_result: dict, blocking: bool = False):
        """
        Memory consolidation: compress → sort → combine.
        MUST be called after Phase B (Part4 feedback) so corrections are stored.
        Can run in a background thread (blocking=False) to avoid delaying UI.
        """
        if blocking:
            self._consolidation_worker(phase_b_result)
        else:
            t = threading.Thread(
                target=self._consolidation_worker,
                args=(phase_b_result,),
                daemon=True
            )
            t.start()
            print("🔄 Phase C (consolidation) started in background thread.")
            return t

    def _should_consolidate(self) -> bool:
        total = self.memory.memory["metadata"]["total_moments"]
        last = self.memory.memory["metadata"].get("last_consolidation")
        return (total % 3 == 0) or (last is None and total > 1)

    def _consolidation_worker(self, phase_b_result: dict):
        """Background consolidation: compress + sort + combine."""
        if not self._should_consolidate():
            enc = self.memory.memory["metadata"]["total_moments"]
            last = self.memory.memory["metadata"].get("last_consolidation")
            print(f"⏭️  Skipping consolidation (moments={enc}, last={last})")
            return

        phase_start = time.time()
        print(f"\n🧹 PHASE C — Memory consolidation (background)")

        memory_json = json.dumps(self.memory.memory, indent=2, ensure_ascii=False)

        # Limit payload size to avoid overwhelming the LLM context
        if len(memory_json) > 8000:
            # Send only the most relevant parts
            trimmed = {
                "layer1": self.memory.memory["layer1"],
                "layer2": self.memory.memory["layer2"][-5:],
                "layer4": self.memory.memory["layer4"],
                "layer5": self.memory.memory["layer5"],
                "layer6": self.memory.memory["layer6"],
                "layer7_indices": {
                    "people": self.memory.memory["layer7"]["people"],
                    "locations": self.memory.memory["layer7"]["locations"],
                    "activity_events": self.memory.memory["layer7"]["activity_events"],
                },
                "metadata": self.memory.memory["metadata"]
            }
            memory_json = json.dumps(trimmed, indent=2, ensure_ascii=False)

        consolidation_prompt = f"""You are a memory consolidation system. Analyse the current memory and:

CURRENT MEMORY:
{memory_json}

TASKS:
1. Compress layers 1-3 into updated summaries for layers 4, 5, 6.
2. Sort layer-7 indices: assign canonical tags to people, locations, activity_events.
3. Combine near-duplicate location / event names into canonical forms.
4. Extract and update user profile (name, preferences, habits) for layer 6.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown:
{{
  "compress": {{
    "layer4": {{
      "summary": "...",
      "current_tasks": ["..."],
      "life_trajectory": "..."
    }},
    "layer5": {{
      "summary": "...",
      "key_events": ["..."],
      "long_term_patterns": "..."
    }},
    "layer6": {{
      "summary": "...",
      "profile": {{
        "name": "...",
        "basic_info": {{}},
        "preferences": {{}},
        "habits": {{}}
      }}
    }}
  }},
  "sort": {{
    "people": {{"canonical_name": ["moment_id_1", "moment_id_2"]}},
    "locations": {{"canonical_location": ["moment_id_1"]}},
    "activity_events": {{"event_tag": ["moment_id_1"]}}
  }},
  "combine": {{
    "locations": {{"old fuzzy name": "canonical name"}},
    "activity_events": {{"old tag": "canonical tag"}}
  }}
}}

IMPORTANT:
- Respond ONLY with valid JSON
- Be aggressive in merging near-duplicate location descriptions
- Layer 4 = recent/current (days); Layer 5 = long-term (months/years); Layer 6 = stable profile"""

        try:
            t0 = time.time()
            resp = requests.post(
                f"{self.qwen_api_url}/consolidate",
                json={"prompt": consolidation_prompt},
                timeout=90
            )
            self.timings_phase_c["llm_consolidation"] = time.time() - t0

            if resp.status_code != 200:
                print(f"❌ Consolidation API error: {resp.status_code}")
                return

            raw = resp.json()["analysis"]
            data = self._parse_json_safe(raw)

            if not data:
                print(f"⚠️  Could not parse consolidation JSON. Preview: {raw[:300]}")
                return

            # Apply operations
            t0 = time.time()
            if "compress" in data:
                self.memory.compress(data["compress"])
            if "sort" in data:
                self.memory.sort(data["sort"])
            if "combine" in data:
                self.memory.combine(data["combine"])
            self.timings_phase_c["apply_operations"] = time.time() - t0

            self.timings_phase_c["total_phase_c"] = time.time() - phase_start
            print(f"✅ Consolidation complete in {self.timings_phase_c['total_phase_c']:.2f}s")
            print(f"\n📋 Updated Memory:\n{self.memory.get_all_memory()}")

        except Exception as e:
            print(f"❌ Consolidation failed: {e}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _parse_json_safe(text: str) -> dict | None:
        """Try several strategies to extract JSON from LLM output."""
        try:
            return json.loads(text)
        except Exception:
            pass
        for pattern in [r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*\})"]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
        return None

    # -----------------------------------------------------------------------
    # ─── Main entry point ────────────────────────────────────────────────
    # -----------------------------------------------------------------------

    def process(self, consolidation_blocking: bool = False) -> dict:
        """
        Full pipeline:
          Phase A (Part1-3) → Phase B (Part4) → Phase C (consolidation, background by default)

        consolidation_blocking=True will wait for Phase C before returning
        (useful for testing; in production keep it False).
        """
        wall_start = time.time()

        # ── Phase A ──────────────────────────────────────────────────────
        phase_a = self.run_phase_a()

        # ── Phase B ──────────────────────────────────────────────────────
        phase_b = self.run_phase_b(phase_a)

        # ── Phase C (after feedback, never blocking Part1-4) ─────────────
        consolidation_thread = self.run_phase_c(phase_b, blocking=consolidation_blocking)

        wall_elapsed = time.time() - wall_start
        self._print_timing_summary(wall_elapsed, consolidation_blocking)

        # Save results
        self._save_results(phase_a, phase_b, wall_elapsed)

        return {
            "phase_a": phase_a,
            "phase_b": phase_b,
            "consolidation_thread": consolidation_thread,
            "timings": {
                "phase_a": self.timings_phase_a,
                "phase_b": self.timings_phase_b,
                "phase_c": self.timings_phase_c,  # may be empty if background
            }
        }

    # -----------------------------------------------------------------------
    # ─── Helpers ─────────────────────────────────────────────────────────
    # -----------------------------------------------------------------------

    def _print_timing_summary(self, wall_elapsed: float, phase_c_included: bool):
        print("\n" + "=" * 60)
        print("⏱️  TIMING SUMMARY")
        print("=" * 60)

        print("\n── Phase A (Part1-3) ──────────────────────────────────────")
        for k, v in sorted(self.timings_phase_a.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k:.<38} {v:>6.2f}s")

        print("\n── Phase B (Part4 AR + Feedback) ──────────────────────────")
        for k, v in sorted(self.timings_phase_b.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k:.<38} {v:>6.2f}s")

        if self.timings_phase_c and phase_c_included:
            print("\n── Phase C (Consolidation) ─────────────────────────────────")
            for k, v in sorted(self.timings_phase_c.items(), key=lambda x: x[1], reverse=True):
                print(f"  {k:.<38} {v:>6.2f}s")
        elif not phase_c_included:
            print("\n── Phase C (Consolidation) ─── running in background thread ──")

        print(f"\n  {'wall_clock_total':.<38} {wall_elapsed:>6.2f}s")
        print("=" * 60 + "\n")

    def _save_results(self, phase_a: dict, phase_b: dict, wall_elapsed: float):
        out = self.output_dir / "analysis_result.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"Device: {self.device}\n")
            if self.device == "cuda":
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"\n{'='*50}\nPHASE A — PART1-3 ANALYSIS\n{'='*50}\n")
            f.write(phase_a["analysis_text"])
            f.write(f"\n\n{'='*50}\nPHASE B — AR PLAN\n{'='*50}\n")
            f.write(json.dumps(phase_b["ar_plan"], indent=2, ensure_ascii=False))
            f.write(f"\n\n{'='*50}\nFINAL MEMORY STATE\n{'='*50}\n")
            f.write(self.memory.get_all_memory())
            f.write(f"\n\n{'='*50}\nTIMINGS\n{'='*50}\n")
            f.write("Phase A:\n")
            for k, v in self.timings_phase_a.items():
                f.write(f"  {k}: {v:.2f}s\n")
            f.write("Phase B:\n")
            for k, v in self.timings_phase_b.items():
                f.write(f"  {k}: {v:.2f}s\n")
            f.write(f"wall_clock_total: {wall_elapsed:.2f}s\n")
        print(f"✅ Results saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    video_path = "/hpc2hdd/home/jyinap/Proactive_Agent/test_data/test_data/test4.mp4"
    assistant = VRAssistant(video_path)

    # blocking=False  → Phase C runs in background (default, best for production)
    # blocking=True   → wait for consolidation before printing final timing
    results = assistant.process(consolidation_blocking=False)