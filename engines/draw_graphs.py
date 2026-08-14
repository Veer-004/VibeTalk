"""
draw_graphs.py — Visualise every LangGraph state machine in VibeTalk.

Outputs four PNG files in the project root:
  • arena_start_graph.png
  • arena_turn_graph.png
  • express_start_graph.png
  • express_turn_graph.png

Run:
    python engines/draw_graphs.py
"""

import os
import sys

# Ensure the project root is on sys.path so `config` and sibling imports work.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import config  # noqa: F401 — load env vars before anything else

from engines.vibe_talk_arena_bot import start_app as arena_start, turn_app as arena_turn
from engines.vibe_talk_express_bot import start_app as express_start, turn_app as express_turn

OUTPUT_DIR = PROJECT_ROOT  # save PNGs at project root; change if you prefer elsewhere


def draw(compiled_graph, filename: str) -> None:
    """Render a compiled LangGraph to a PNG file."""
    out_path = os.path.join(OUTPUT_DIR, filename)
    try:
        # .get_graph().draw_mermaid_png() returns PNG bytes
        png_bytes = compiled_graph.get_graph().draw_mermaid_png()
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"  ✅  {filename}  →  {out_path}")
    except Exception as e:
        print(f"  ❌  {filename}  →  {e}")


def main():
    print("\n🎓 VibeTalk — Drawing LangGraph state machines\n")

    graphs = [
        (arena_start,   "arena_start_graph.png"),
        (arena_turn,    "arena_turn_graph.png"),
        (express_start, "express_start_graph.png"),
        (express_turn,  "express_turn_graph.png"),
    ]

    for graph, name in graphs:
        draw(graph, name)

    print("\nDone! Open the PNG files to see your graphs.\n")


if __name__ == "__main__":
    main()
