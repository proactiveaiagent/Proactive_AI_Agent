#!/usr/bin/env python3
"""Explicitly synchronize a GTE sidecar index from one PersonMemory JSON store."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from semantic_retrieval import GTEEncoder, PersistentSemanticIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory_dir", type=Path,
                        help="Directory containing memory.json; index stays in this directory")
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    memory_path = args.memory_dir / "memory.json"
    if not memory_path.is_file():
        parser.error(f"No memory.json at {memory_path}")
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    moments = memory.get("layer7", {}).get("moments", {})
    encoder = GTEEncoder(args.model_root)
    index = PersistentSemanticIndex(args.memory_dir)
    print(json.dumps(index.sync(moments, encoder), ensure_ascii=False, indent=2))
    print(json.dumps(index.stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
