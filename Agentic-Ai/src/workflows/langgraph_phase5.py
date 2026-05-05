"""
langgraph_phase5.py  –  Phase 5 LangGraph Workflow (Intelligent Edit & Undo)

Pipeline
--------
classify_intent → plan_edit → snapshot_state → execute_edit → update_state → END

The workflow receives a free-text edit query, classifies the intent via LLM,
snapshots the current state, executes the edit by dispatching to the correct
pipeline phase, and updates the working state.

Uses LangGraph's StateGraph (same pattern as Phase 1–3 workflows).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.agents import edit_executor, edit_intent_classifier
from src.agents.edit_intent_classifier import EditIntent
from src.state_versioning import StateManager
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


# ── State ────────────────────────────────────────────────────────────────────

class Phase5State(TypedDict, total=False):
    # Input
    edit_query: str
    run_config: Dict[str, Any]
    tool_client: object

    # Internal
    intent: Dict[str, Any]          # serialised EditIntent
    edit_plan: Dict[str, Any]
    snapshot_version: int
    execution_result: Dict[str, Any]

    # Output
    phase5_state: Dict[str, Any]    # status, result summary
    errors: List[str]

    # Infrastructure
    state_manager: object           # StateManager instance


# ── Nodes ────────────────────────────────────────────────────────────────────

def classify_intent_node(state: Phase5State) -> Phase5State:
    """
    Classify the user's free-text edit query into a structured intent.
    Uses LLM (via tool_client) with rule-based fallback.
    """
    print("\n" + "=" * 60)
    print("  PHASE 5 — Intelligent Edit & Undo")
    print("=" * 60)

    query = state.get("edit_query", "").strip()
    if not query:
        state.setdefault("errors", []).append("Empty edit query")
        state["phase5_state"] = {"status": "failed", "error": "Empty edit query"}
        return state

    print(f"[Phase 5] classify_intent: query = {query!r}")

    tool_client = state.get("tool_client")

    # Load current context for better classification
    context = None
    try:
        current_state = edit_executor.collect_current_state()
        scene_manifest = current_state.get("scene_manifest", {})
        character_db = current_state.get("character_db", {})
        context = {
            "scenes": scene_manifest.get("scenes", []),
            "characters": character_db.get("characters", []),
        }
    except Exception:
        pass

    if tool_client:
        try:
            intent = edit_intent_classifier.classify(
                query, tool_client, context=context
            )
        except Exception as exc:
            LOGGER.warning("LLM classify failed, using rule-based: %s", exc)
            intent = edit_intent_classifier.classify_without_llm(query)
    else:
        intent = edit_intent_classifier.classify_without_llm(query)

    state["intent"] = intent.model_dump()
    state.setdefault("phase5_state", {})["intent"] = intent.model_dump()

    print(f"[Phase 5] classify_intent: intent={intent.intent}, "
          f"target={intent.target}, scope={intent.scope}, "
          f"confidence={intent.confidence:.2f}")
    print(f"[Phase 5] classify_intent: parameters={intent.parameters}")

    return state


def plan_edit_node(state: Phase5State) -> Phase5State:
    """
    Validate the intent and create an execution plan.
    If confidence is too low, flag for clarification.
    """
    print("\n[Phase 5] plan_edit: creating execution plan...")

    intent_data = state.get("intent", {})
    intent = EditIntent(**intent_data)

    plan: Dict[str, Any] = {
        "intent": intent.intent,
        "target": intent.target,
        "scope": intent.scope,
        "parameters": intent.parameters,
        "confidence": intent.confidence,
        "needs_clarification": False,
        "phases_to_rerun": [],
    }

    # Low confidence → flag for clarification
    if intent.confidence < 0.5:
        plan["needs_clarification"] = True
        plan["clarification_message"] = (
            f"I'm not confident about this edit (confidence: {intent.confidence:.0%}). "
            f"Did you mean to {intent.intent} targeting {intent.target}?"
        )
        print(f"[Phase 5] plan_edit: LOW confidence ({intent.confidence:.2f}), "
              "flagging for clarification")
    else:
        # Determine which phases to re-run
        if intent.target == "script":
            plan["phases_to_rerun"] = ["phase1", "phase2", "phase3"]
        elif intent.target == "audio":
            plan["phases_to_rerun"] = ["phase2", "phase3"]
        elif intent.target == "video_frame":
            if intent.intent == "apply_filter" or "brightness" in intent.intent.lower():
                plan["phases_to_rerun"] = ["filter", "phase3"]
            else:
                plan["phases_to_rerun"] = ["phase3"]
        elif intent.target == "video":
            plan["phases_to_rerun"] = ["phase3"]

        print(f"[Phase 5] plan_edit: will re-run {plan['phases_to_rerun']}")

    state["edit_plan"] = plan
    state.setdefault("phase5_state", {})["plan"] = plan
    return state


def snapshot_state_node(state: Phase5State) -> Phase5State:
    """
    Snapshot the current pipeline state before making changes.
    This enables undo/revert functionality.
    """
    print("\n[Phase 5] snapshot_state: saving current state...")

    plan = state.get("edit_plan", {})

    # Skip snapshot if we need clarification
    if plan.get("needs_clarification"):
        print("[Phase 5] snapshot_state: skipped (needs clarification)")
        return state

    sm: StateManager = state.get("state_manager")  # type: ignore[assignment]
    if sm is None:
        sm = StateManager()

    # Collect current state
    current_state = edit_executor.collect_current_state()
    asset_paths = edit_executor.collect_current_asset_paths()

    intent_data = state.get("intent", {})
    description = (
        f"Before edit: {intent_data.get('intent', 'unknown')} "
        f"(target: {intent_data.get('target', 'unknown')})"
    )

    # Only snapshot if this is the first version or state has changed
    version = sm.current_version()
    if version == 0:
        # First snapshot: save as initial pipeline output
        version = sm.snapshot(
            state_json=current_state,
            asset_paths=asset_paths,
            description="Initial pipeline output",
            target="pipeline",
        )
        print(f"[Phase 5] snapshot_state: initial snapshot saved as v{version}")

    # Now snapshot the pre-edit state
    version = sm.snapshot(
        state_json=current_state,
        asset_paths=asset_paths,
        description=description,
        target=intent_data.get("target", "pipeline"),
    )

    state["snapshot_version"] = version
    state["state_manager"] = sm  # type: ignore[typeddict-item]
    state.setdefault("phase5_state", {})["snapshot_version"] = version

    print(f"[Phase 5] snapshot_state: saved as v{version}")
    return state


def execute_edit_node(state: Phase5State) -> Phase5State:
    """
    Execute the edit by dispatching to the correct pipeline phase.
    """
    plan = state.get("edit_plan", {})

    # Skip execution if we need clarification
    if plan.get("needs_clarification"):
        state["execution_result"] = {
            "success": False,
            "description": plan.get("clarification_message", "Clarification needed"),
            "target": plan.get("target", "unknown"),
            "changes": [],
            "errors": ["Clarification needed from user"],
            "needs_clarification": True,
        }
        state.setdefault("phase5_state", {})["status"] = "needs_clarification"
        print(f"[Phase 5] execute_edit: skipped — {plan.get('clarification_message')}")
        return state

    print("\n[Phase 5] execute_edit: executing...")

    intent_data = state.get("intent", {})
    intent = EditIntent(**intent_data)
    current_state = edit_executor.collect_current_state()
    config = state.get("run_config", {})

    result = edit_executor.execute(intent, current_state, config)

    state["execution_result"] = result
    state.setdefault("phase5_state", {})["execution_result"] = result

    status = "completed" if result.get("success") else "failed"
    print(f"[Phase 5] execute_edit: {status}")
    for change in result.get("changes", []):
        print(f"  - {change}")
    for error in result.get("errors", []):
        print(f"  ! {error}")

    return state


def update_state_node(state: Phase5State) -> Phase5State:
    """
    Update the working state files and create a post-edit snapshot.
    """
    print("\n[Phase 5] update_state: finalising...")

    result = state.get("execution_result", {})
    plan = state.get("edit_plan", {})

    if plan.get("needs_clarification"):
        state.setdefault("phase5_state", {})["status"] = "needs_clarification"
        return state

    sm: StateManager = state.get("state_manager")  # type: ignore[assignment]
    if sm is None:
        sm = StateManager()

    # If edit succeeded, snapshot the new state
    if result.get("success"):
        new_state = edit_executor.collect_current_state()
        new_assets = edit_executor.collect_current_asset_paths()
        intent_data = state.get("intent", {})
        description = (
            f"After edit: {intent_data.get('intent', 'unknown')} — "
            + "; ".join(result.get("changes", ["no changes"]))
        )
        version = sm.snapshot(
            state_json=new_state,
            asset_paths=new_assets,
            description=description,
            target=intent_data.get("target", "pipeline"),
        )
        state.setdefault("phase5_state", {})["new_version"] = version
        print(f"[Phase 5] update_state: post-edit snapshot saved as v{version}")
    else:
        print("[Phase 5] update_state: edit failed, no new snapshot")

    state.setdefault("phase5_state", {})["status"] = (
        "completed" if result.get("success") else "failed"
    )
    state["phase5_state"]["result_summary"] = {
        "success": result.get("success", False),
        "description": result.get("description", ""),
        "changes": result.get("changes", []),
        "errors": result.get("errors", []),
    }

    print(f"[Phase 5] Done.")
    return state


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the Phase 5 LangGraph workflow."""
    graph = StateGraph(Phase5State)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("plan_edit",       plan_edit_node)
    graph.add_node("snapshot_state",  snapshot_state_node)
    graph.add_node("execute_edit",    execute_edit_node)
    graph.add_node("update_state",    update_state_node)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "plan_edit")
    graph.add_edge("plan_edit",       "snapshot_state")
    graph.add_edge("snapshot_state",  "execute_edit")
    graph.add_edge("execute_edit",    "update_state")
    graph.add_edge("update_state",    END)

    return graph.compile()


# ── Convenience runner ───────────────────────────────────────────────────────

def run_edit(
    query: str,
    tool_client: object = None,
    run_config: Optional[Dict[str, Any]] = None,
    state_manager: Optional[StateManager] = None,
) -> Dict[str, Any]:
    """
    Run the full Phase 5 edit workflow for a single query.

    Parameters
    ----------
    query : str
        Free-text edit command from the user.
    tool_client : object, optional
        MCP ToolClient for LLM access.
    run_config : dict, optional
        Pipeline configuration (quality, backend, etc.)
    state_manager : StateManager, optional
        Override the default StateManager instance.

    Returns
    -------
    dict
        The final phase5_state with status, intent, result summary.
    """
    state: Phase5State = {
        "edit_query": query,
        "tool_client": tool_client,
        "run_config": run_config or {},
        "state_manager": state_manager or StateManager(),
        "errors": [],
        "phase5_state": {"status": "processing"},
    }

    graph = build_graph()
    result = graph.invoke(state)

    return result.get("phase5_state", {})
