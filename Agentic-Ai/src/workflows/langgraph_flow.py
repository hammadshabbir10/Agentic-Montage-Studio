from typing import Dict, TypedDict
from langgraph.graph import END, StateGraph
from src.agents import character_designer, hitl, image_synthesizer, scriptwriter, validator


class GraphState(TypedDict, total=False):
    mode: str
    prompt: str
    script_text: str
    manifest: Dict[str, object]
    story_manifest: Dict[str, object]      # NEW — story JSON envelope
    character_db: Dict[str, object]
    image_assets: list
    script_state: Dict[str, object]
    approved: bool
    errors: list
    tool_client: object
    memory_store: object
    auto_approve: bool
    num_scenes: int


def mode_selector_node(state: GraphState) -> GraphState:
    if "script_state" not in state:
        state["script_state"] = {
            "input_mode": state.get("mode", ""),
            "script": {},
            "story": {},
            "characters": [],
            "images": [],
            "status": "processing",
        }
    return state


# ---------------------------------------------------------------------------
# Story node  (NEW)
# ---------------------------------------------------------------------------

def story_node(state: GraphState) -> GraphState:
    """
    Generates the story manifest.

    - auto  mode: derives story from the user prompt
    - manual mode: infers story from the already-loaded script_text
    """
    from src.agents import story_generator          # placed in src/agents/
    from src.io.json_schema import build_story_manifest

    mode = state.get("mode", "auto")
    tool_client = state["tool_client"]

    if mode == "manual":
        raw_story = story_generator.run_from_script(
            state.get("script_text", ""), tool_client
        )
    else:
        raw_story = story_generator.run_from_prompt(
            state.get("prompt", ""),
            state.get("num_scenes", 4),
            tool_client,
        )

    story_manifest = build_story_manifest(raw_story)
    state["story_manifest"] = story_manifest

    if state.get("script_state") is not None:
        state["script_state"]["story"] = story_manifest.get("story", {})

    return state


# ---------------------------------------------------------------------------
# Existing nodes (unchanged logic, story_state key added where sensible)
# ---------------------------------------------------------------------------

def validator_node(state: GraphState) -> GraphState:
    result = validator.run(state.get("script_text", ""))
    state.update({"errors": result["errors"], "manifest": result.get("manifest", {})})
    if state.get("script_state") is not None:
        state["script_state"]["script"] = state.get("manifest", {})
    return state


def scriptwriter_node(state: GraphState) -> GraphState:
    from src.io.json_schema import build_scene_manifest
    result = scriptwriter.run(state.get("prompt", ""), state["tool_client"])
    manifest = (
        build_scene_manifest(result["manifest"].get("scenes", []))
        if "scenes" in result["manifest"]
        else result["manifest"]
    )
    state.update({"script_text": result["script_text"], "manifest": manifest})
    if state.get("script_state") is not None:
        state["script_state"]["script"] = state.get("manifest", {})
    return state


def hitl_node(state: GraphState) -> GraphState:
    if state.get("errors"):
        state["approved"] = False
        if state.get("script_state") is not None:
            state["script_state"]["status"] = "rejected"
        return state
    state["approved"] = hitl.run(state.get("auto_approve", False))
    if state.get("script_state") is not None:
        state["script_state"]["status"] = "approved" if state.get("approved") else "rejected"
    return state


def character_node(state: GraphState) -> GraphState:
    from src.io.json_schema import build_character_db
    characters = character_designer.run(state.get("manifest", {}), state["tool_client"])
    character_db = build_character_db(characters)
    state["character_db"] = character_db
    if state.get("script_state") is not None:
        state["script_state"]["characters"] = characters
    return state


def image_node(state: GraphState) -> GraphState:
    from src.io.json_schema import build_character_db
    characters = state.get("character_db", {}).get("characters", [])
    character_db = build_character_db(characters)
    state["character_db"] = character_db
    return state


def memory_commit_node(state: GraphState) -> GraphState:
    payload = {
        "story":      state.get("story_manifest", {}),    # NEW
        "manifest":   state.get("manifest", {}),
        "characters": state.get("character_db", {}),
        "script_state": state.get("script_state", {}),
    }
    state["tool_client"].invoke_by_capability("commit_memory", payload)
    if state.get("script_state") is not None:
        state["script_state"]["status"] = "completed"
    return state


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("mode_selector",  mode_selector_node)
    graph.add_node("story",          story_node)           # NEW
    graph.add_node("validator",      validator_node)
    graph.add_node("scriptwriter",   scriptwriter_node)
    graph.add_node("hitl",           hitl_node)
    graph.add_node("character",      character_node)
    graph.add_node("image",          image_node)
    graph.add_node("memory_commit",  memory_commit_node)

    graph.set_entry_point("mode_selector")

    # mode_selector → story (always, for both modes)
    graph.add_edge("mode_selector", "story")

    # story → validator (manual) or scriptwriter (auto)
    graph.add_conditional_edges(
        "story",
        lambda state: "validator" if state.get("mode") == "manual" else "scriptwriter",
        {"validator": "validator", "scriptwriter": "scriptwriter"},
    )

    graph.add_edge("validator",    "hitl")
    graph.add_edge("scriptwriter", "hitl")

    graph.add_conditional_edges(
        "hitl",
        lambda state: "character" if state.get("approved") else END,
        {"character": "character", END: END},
    )

    graph.add_edge("character",     "image")
    graph.add_edge("image",         "memory_commit")
    graph.add_edge("memory_commit", END)

    return graph.compile()