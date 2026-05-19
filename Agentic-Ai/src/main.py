import argparse
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from pydantic import ValidationError
from src.io.script_ingest import parse_script_to_manifest
from src.io.consistency import enforce_phase1_character_consistency
from src.io.pydantic_schemas import validate_phase1_payloads
from src.mcp.tool_client import ToolClient
from src.mcp.tool_registry import ToolRegistry
from src.memory.vector_store import MemoryStore
from src.workflows.langgraph_flow import build_graph


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_PHASE1_COUNTER_FILE = Path("data/phase1_run_counter.json")
_PHASE1_RUNS_DIR = Path("data/phase1_runs")


def _read_phase1_counter() -> int:
    try:
        data = json.loads(_PHASE1_COUNTER_FILE.read_text(encoding="utf-8"))
        return int(data.get("last_run", 0))
    except Exception:
        return 0


def _write_phase1_counter(n: int) -> None:
    _PHASE1_COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PHASE1_COUNTER_FILE.write_text(
        json.dumps({"last_run": n}, indent=2), encoding="utf-8"
    )


def _get_next_phase1_run() -> dict:
    n = _read_phase1_counter() + 1
    run_tag = f"run{n:02d}"
    run_dir = _PHASE1_RUNS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_phase1_counter(n)
    return {"run_number": n, "run_tag": run_tag, "run_dir": run_dir}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Project Montage Phase 1")
    parser.add_argument("--mode", choices=["manual", "auto"], required=True)
    parser.add_argument("--script-path", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument(
        "--scenes", type=int, default=0,
        help="Number of scenes to generate (auto mode)",
    )
    parser.add_argument("--auto-approve", action="store_true")
    args = parser.parse_args()

    registry = ToolRegistry()
    memory_store = MemoryStore(persist_dir="data/memory")
    tool_client = ToolClient(
        registry, memory_store=memory_store, image_dir="data/image_assets"
    )

    short_prompt = args.prompt.strip()

    # Determine num_scenes: CLI flag > embedded in prompt > ask once
    num_scenes = args.scenes
    if num_scenes <= 0:
        match = re.search(r"(\d+)\s*scenes?", short_prompt, re.IGNORECASE)
        if match:
            num_scenes = int(match.group(1))
        elif args.mode == "auto":
            try:
                raw = input("Enter number of scenes (default 4): ").strip()
                num_scenes = int(raw) if raw else 4
            except Exception:
                num_scenes = 4

    # Build a rich backend prompt that tells the LLM exactly what structure to emit
    backend_prompt = (
        f"{short_prompt}. "
        f"Write exactly {num_scenes} scenes. "
        "Format each scene as a standard screenplay: "
        "start with a scene heading (INT. or EXT. LOCATION - DAY/NIGHT), "
        "followed by action lines describing the setting and atmosphere, "
        "then dialogue blocks using the format CHARACTER NAME: \"dialogue line\" "
        "with a VISUAL CUE line immediately after each dialogue block describing the "
        "specific shot (e.g. 'VISUAL CUE: Close-up of CHARACTER, tension visible on face.'). "
        "Do NOT use numbered headings like '1:' or '2:' — use only standard INT./EXT. headings. "
        "Every character must have a unique, specific visual cue per dialogue line. "
        "Never write 'Default visual cue'. "
        "Ensure each character has a distinct personality visible through their dialogue. "
        "Characters introduced must stay consistent throughout all scenes. "
        "IMPORTANT: All female character names must end with the letter 'A'. "
        "All male character names must NOT end with the letter 'A'."
    )

    # Hardcoded Car/Racing awareness
    if any(word in short_prompt.lower() for word in ["car", "race", "racing", "drive", "driving", "vehicle", "hadi", "haider"]):
        backend_prompt += (
            " \nCRITICAL VISUAL REQUIREMENT: Since this is a racing/car-themed story, ensure that the high-performance cars "
            "are prominently visible in the foreground of every action and racing scene. Descriptions must include 'deep focus' "
            "and 'sharp backgrounds' to ensure both the characters and their cars are clearly visible without background blur. "
            "Identify the cars by color (e.g., 'Hadi's sleek blue car', 'Haider's rugged red car') in the VISUAL CUES."
        )

    state = {
        "mode":       args.mode,
        "script_text": "",
        "prompt":     backend_prompt,
        "num_scenes": num_scenes,
        "manifest":   {},
        "story_manifest": {},          # NEW
        "character_db": {},
        "errors":     [],
        "script_state": {
            "input_mode": args.mode,
            "story":      {},          # NEW
            "script":     {},
            "characters": [],
            "images":     [],
            "status":     "processing",
        },
        "auto_approve": args.auto_approve,
        "tool_client":  tool_client,
        "memory_store": memory_store,
    }

    if args.mode == "manual":
        if not args.script_path:
            raise SystemExit("--script-path is required for manual mode")
        script_text = Path(args.script_path).read_text(encoding="utf-8")
        state["script_text"] = script_text
        state["manifest"] = parse_script_to_manifest(script_text, title="Manual Script")

    graph = build_graph()
    result = graph.invoke(state)

    if result.get("errors"):
        print("Validation errors:")
        for error in result["errors"]:
            print(f"  - {error}")
        raise SystemExit(1)

    if not result.get("approved"):
        print("Script not approved. Exiting.")
        raise SystemExit(1)

    story_manifest = result.get("story_manifest", {})
    scene_manifest = result.get("manifest", {})
    character_db = result.get("character_db", {})

    # Reduce drift: keep story protagonist/antagonist aligned to actual scene characters.
    story_manifest, scene_manifest, character_db, consistency_warnings = (
        enforce_phase1_character_consistency(
            story_manifest=story_manifest,
            scene_manifest=scene_manifest,
            character_db=character_db,
        )
    )
    if consistency_warnings:
        print("Consistency adjustments:")
        for warning in consistency_warnings:
            print(f"  - {warning}")

    # Strict schema enforcement with Pydantic before writing outputs.
    try:
        story_manifest, scene_manifest, character_db = validate_phase1_payloads(
            story_manifest=story_manifest,
            scene_manifest=scene_manifest,
            character_db=character_db,
        )
    except ValidationError as exc:
        print("Phase 1 schema validation failed:")
        print(exc)
        raise SystemExit(2)

    mode_suffix = args.mode
    run_info = _get_next_phase1_run()
    run_dir = run_info["run_dir"]

    # --- Save outputs ---
    _write_json(
        Path(f"data/story_manifest_{mode_suffix}.json"),
        story_manifest,
    )
    _write_json(
        Path(f"data/scene_manifest_{mode_suffix}.json"),
        scene_manifest,
    )
    _write_json(
        Path(f"data/character_db_{mode_suffix}.json"),
        character_db,
    )
    if result.get("script_text"):
        _write_text(
            Path(f"data/last_script_{mode_suffix}.txt"),
            result["script_text"],
        )

    # Also save a per-run snapshot in data/phase1_runs/runXX
    _write_json(run_dir / "story_manifest.json", story_manifest)
    _write_json(run_dir / "scene_manifest.json", scene_manifest)
    _write_json(run_dir / "character_db.json", character_db)
    if result.get("script_text"):
        _write_text(run_dir / "last_script.txt", result["script_text"])

    print(
        f"Outputs saved:\n"
        f"  data/story_manifest_{mode_suffix}.json\n"
        f"  data/scene_manifest_{mode_suffix}.json\n"
        f"  data/character_db_{mode_suffix}.json\n"
        f"  data/phase1_runs/{run_info['run_tag']}/story_manifest.json\n"
        f"  data/phase1_runs/{run_info['run_tag']}/scene_manifest.json\n"
        f"  data/phase1_runs/{run_info['run_tag']}/character_db.json"
    )


if __name__ == "__main__":
    main()