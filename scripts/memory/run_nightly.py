"""CLI for running end-of-day memory consolidation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from memory import PersonMemory
from nighttime_processor import NightlyProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-dir", default=str(CODE / "memory"),
        help="Directory containing memory.json (default: code/memory)",
    )
    parser.add_argument("--day", help="ISO date YYYY-MM-DD; defaults to memory metadata.today")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Qwen API base URL")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even when this day's source digest is unchanged")
    args = parser.parse_args()

    memory = PersonMemory(memory_dir=args.memory_dir)
    result = NightlyProcessor(memory, qwen_api_url=args.api_url).run(
        day=args.day, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
