#!/usr/bin/env python3
"""Embed maintained visualization modules into the single-file server page."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
MODULES = {
    "evidence_gallery.css": ("/* GENERATED MODULE evidence_gallery.css:START */",
                             "/* GENERATED MODULE evidence_gallery.css:END */"),
    "evidence_gallery.js": ("/* GENERATED MODULE evidence_gallery.js:START */",
                            "/* GENERATED MODULE evidence_gallery.js:END */"),
}

def render():
    text = INDEX.read_text(encoding="utf-8")
    for name, (start, end) in MODULES.items():
        module = (ROOT / "modules" / name).read_text(encoding="utf-8").rstrip()
        begin = text.find(start)
        finish = text.find(end, begin + len(start)) if begin >= 0 else -1
        if begin < 0 or finish < 0:
            raise SystemExit(f"missing module markers for {name}")
        content_start = begin + len(start)
        text = text[:content_start] + chr(10) + module + chr(10) + text[finish:]
    return text

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = render()
    if args.check:
        if INDEX.read_text(encoding="utf-8") != output:
            raise SystemExit("index.html is stale; run build_viz.py")
        print("visualization bundle is current")
        return
    INDEX.write_text(output, encoding="utf-8")
    print("built index.html from visualization modules")

if __name__ == "__main__":
    main()
