"""
main_phase2.py  –  Phase 2 Entry Point

Usage
-----
  python main_phase2.py --manifest data/scene_manifest_auto.json \
                        --characters data/character_db_auto.json

  # or use the defaults (auto mode outputs from Phase 1):
  python main_phase2.py
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.mcp.tool_client import ToolClient
from src.mcp.tool_registry import ToolRegistry
from src.memory.vector_store import MemoryStore
from src.run_manager import get_next_run, save_task_graph_log
from src.workflows.langgraph_phase2 import build_graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_banner(title: str, width: int = 60) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Project Montage – Phase 2: Audio")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/scene_manifest_auto.json",
        help="Path to scene_manifest JSON from Phase 1",
    )
    parser.add_argument(
        "--characters",
        type=str,
        default="data/character_db_auto.json",
        help="Path to character_db JSON from Phase 1",
    )
    args = parser.parse_args()

    # ── Load Phase 1 outputs ─────────────────────────────────────────────────
    _print_banner("PHASE 2 — Audio Generation & Integration  [STARTING]")
    print(f"[Phase 2] Loading manifest   : {args.manifest}")
    print(f"[Phase 2] Loading characters : {args.characters}")

    manifest = _load_json(args.manifest)
    char_db  = _load_json(args.characters)
    scenes   = manifest.get("scenes", [])

    if not scenes:
        print("[Phase 2] ERROR: No scenes found in manifest. Exiting.")
        raise SystemExit(1)

    print(f"[Phase 2] Scenes loaded      : {len(scenes)}")
    print(f"[Phase 2] Characters loaded  : {char_db.get('total_characters', '?')}")

    # ── Run directory setup ──────────────────────────────────────────────────
    scene_ids = [s["scene_id"] for s in scenes]
    run_info  = get_next_run(scene_ids)
    run_tag   = run_info["run_tag"]
    run_dir   = run_info["run_dir"]

    print(f"\n[Phase 2] Run tag  : {run_tag.upper()}")
    print(f"[Phase 2] Run dir  : {run_dir}")

    save_task_graph_log(
        log_path=run_info["log_path"],
        run_info=run_info,
        scene_manifest=manifest,
        character_db=char_db,
    )

    # ── Infrastructure ───────────────────────────────────────────────────────
    registry     = ToolRegistry()
    memory_store = MemoryStore(persist_dir="data/memory")
    tool_client  = ToolClient(
        registry,
        memory_store=memory_store,
        image_dir="data/image_assets",
    )

    output_dirs = {
        "audio":  str(run_dir / "audio"),
        "bgm":    str(run_dir / "bgm"),
        "frames": str(run_dir / "frames"),
    }
    for d in output_dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Build and run LangGraph ──────────────────────────────────────────────
    state = {
        "manifest":    manifest,
        "task_graph":  [],
        "audio_results": [],
        "music_results": [],
        "errors":      [],
        "tool_client": tool_client,
        "memory_store": memory_store,
        "phase2_state": {"status": "processing"},
        "run_tag":     run_tag,
        "run_dir":     str(run_dir),
        "output_dirs": output_dirs,
    }

    graph  = build_graph()
    result = graph.invoke(state)

    # ── Save outputs ─────────────────────────────────────────────────────────
    _write_json(
        run_dir / f"task_graph_{run_tag}.json",
        {"tasks": result.get("task_graph", [])},
    )
    _write_json(
        run_dir / f"phase2_state_{run_tag}.json",
        result.get("phase2_state", {}),
    )
    _write_json(
        run_dir / f"phase2_outputs_{run_tag}.json",
        {
            "audio":  result.get("audio_results", []),
            "music":  result.get("music_results", []),
        },
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    timing_path = result.get("timing_manifest_path", "N/A")
    audio_files = result.get("audio_results", [])

    _print_banner(f"PHASE 2 — COMPLETE  [{run_tag.upper()}]")
    print(f"  Scenes processed : {len(audio_files)}")
    print(f"  Audio folder     : {output_dirs['audio']}")
    print(f"  BGM folder       : {output_dirs['bgm']}")
    print(f"  Timing manifest  : {timing_path}")
    print(f"  Task graph log   : {run_info['log_path']}")

    # Show BGM source summary
    music_results = result.get("music_results", [])
    if music_results:
        print("\n  BGM sources:")
        for m in music_results:
            src = m.get("bgm_source", "?")
            icon = {"local": "📁", "freesound": "🌐", "musicgen": "🎵", "stub": "🔇"}.get(src, "?")
            print(f"    Scene {m['scene_id']:02d} [{m['mood']:12s}] {icon} {src}  → {m['bgm_path']}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()