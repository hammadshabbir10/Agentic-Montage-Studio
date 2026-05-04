"""
langgraph_phase2.py  –  Phase 2 LangGraph workflow (Audio + Music)

Pipeline
--------
scene_parser → voice_synth → music_select → timing_build → memory_commit → END

Key change: global_voice_map is built ONCE in scene_parser_node so every
character has the same voice across all scenes.
"""

from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.agents import music_selector, voice_synthesizer
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────
class GraphState(TypedDict, total=False):
    manifest:             Dict
    task_graph:           List[Dict]
    global_voice_map:     Dict[str, str]   # NEW – character → voice, stable across scenes
    audio_results:        List[Dict]
    music_results:        List[Dict]
    timing_manifest_path: str
    errors:               List[str]
    tool_client:          object
    memory_store:         object
    phase2_state:         Dict
    run_tag:              str
    run_dir:              str
    output_dirs:          Dict[str, str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def scene_parser_node(state: GraphState) -> GraphState:
    print("\n" + "=" * 60)
    print("  PHASE 2 — Audio Generation & Integration")
    print("=" * 60)
    print("[Phase 2] scene_parser: extracting scene tasks...")

    manifest = state.get("manifest", {})
    scenes   = manifest.get("scenes", [])

    tasks = [
        {
            "scene_id":   scene.get("scene_id", i + 1),
            "location":   scene.get("location", ""),
            "dialogue":   scene.get("dialogue", []),
            "characters": scene.get("characters", []),
        }
        for i, scene in enumerate(scenes)
    ]

    # Build global voice map across ALL scenes at once
    global_voice_map = voice_synthesizer.build_global_voice_map(manifest)

    print(f"[Phase 2] scene_parser: {len(tasks)} scene(s) found.")
    print("[Phase 2] scene_parser: global voice assignments:")
    for name, voice in global_voice_map.items():
        gender = "F" if voice_synthesizer._is_female(name) else "M"
        print(f"           [{gender}] {name:40s} -> {voice}")

    state["task_graph"]       = tasks
    state["global_voice_map"] = global_voice_map
    state["audio_results"]    = []
    state["music_results"]    = []
    state["phase2_state"]     = {"status": "processing", "audio": [], "bgm": []}
    return state


def voice_synth_node(state: GraphState) -> GraphState:
    print("\n[Phase 2] -- Voice Synthesis ------------------------------")
    output_dirs      = state.get("output_dirs", {})
    audio_dir        = output_dirs.get("audio", "data/audio")
    run_tag          = state.get("run_tag")
    global_voice_map = state.get("global_voice_map", {})

    audio_results: List[Dict] = []

    for task in state.get("task_graph", []):
        sid = task["scene_id"]
        print(f"[Phase 2] voice_synth: processing scene {sid}...")
        result = voice_synthesizer.run(
            task,
            state["tool_client"],
            audio_dir=audio_dir,
            run_tag=run_tag,
            global_voice_map=global_voice_map,   # pass global map
        )
        audio_results.append(result)

    state["audio_results"] = audio_results
    if state.get("phase2_state"):
        state["phase2_state"]["audio"] = [r["path"] for r in audio_results]

    print(f"[Phase 2] voice_synth: all {len(audio_results)} scene(s) synthesised.")
    return state


def music_select_node(state: GraphState) -> GraphState:
    print("\n[Phase 2] -- BGM Selection ---------------------------------")
    output_dirs = state.get("output_dirs", {})
    run_dir     = state.get("run_dir", output_dirs.get("audio", "data/audio"))
    bgm_out_dir = str(run_dir) + "/bgm" if run_dir else "data/audio/bgm"

    music_results: List[Dict] = []

    for task in state.get("task_graph", []):
        sid = task["scene_id"]
        print(f"[Phase 2] music_select: scene {sid}…")
        result = music_selector.run(
            task,
            bgm_library_dir="data/bgm_library",
            audio_dir=bgm_out_dir,
            bgm_dir=bgm_out_dir,
        )
        music_results.append(result)
        print(
            f"[Phase 2] music_select: scene {sid} "
            f"mood={result['mood']!r} source={result.get('bgm_source','?')!r} "
            f"-> {result['bgm_path']}"
        )

    state["music_results"] = music_results
    if state.get("phase2_state"):
        state["phase2_state"]["bgm"] = [r["bgm_path"] for r in music_results]

    return state


def timing_build_node(state: GraphState) -> GraphState:
    from src.utils.timing_manifest import build as build_timing

    print("\n[Phase 2] -- Timing Manifest -------------------------------")
    output_dirs = state.get("output_dirs", {})
    run_tag     = state.get("run_tag", "run00")
    out_dir     = state.get("run_dir", output_dirs.get("audio", "data/audio"))

    timing_path = build_timing(
        audio_results=state.get("audio_results", []),
        music_results=state.get("music_results", []),
        run_tag=run_tag,
        out_dir=out_dir,
    )
    state["timing_manifest_path"] = timing_path
    print(f"[Phase 2] timing_build: manifest written -> {timing_path}")
    return state


def memory_commit_node(state: GraphState) -> GraphState:
    print("\n[Phase 2] -- Memory Commit ---------------------------------")
    payload = {
        "manifest":             state.get("manifest", {}),
        "audio_results":        state.get("audio_results", []),
        "music_results":        state.get("music_results", []),
        "timing_manifest_path": state.get("timing_manifest_path", ""),
        "phase2_state":         state.get("phase2_state", {}),
    }
    try:
        state["tool_client"].invoke_by_capability("commit_memory", payload)
        print("[Phase 2] memory_commit: done.")
    except Exception as exc:
        LOGGER.warning("memory_commit failed (non-fatal): %s", exc)
        print(f"[Phase 2] memory_commit: skipped ({exc})")

    if state.get("phase2_state"):
        state["phase2_state"]["status"] = "completed"
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("scene_parser",  scene_parser_node)
    graph.add_node("voice_synth",   voice_synth_node)
    graph.add_node("music_select",  music_select_node)
    graph.add_node("timing_build",  timing_build_node)
    graph.add_node("memory_commit", memory_commit_node)

    graph.set_entry_point("scene_parser")
    graph.add_edge("scene_parser",  "voice_synth")
    graph.add_edge("voice_synth",   "music_select")
    graph.add_edge("music_select",  "timing_build")
    graph.add_edge("timing_build",  "memory_commit")
    graph.add_edge("memory_commit", END)

    return graph.compile()