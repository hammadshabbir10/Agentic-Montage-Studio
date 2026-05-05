"""
edit_agent/planner.py  –  Re-export wrapper for Phase 5 Edit Planner

The planning logic is integrated into the LangGraph workflow's plan_edit node.
See Agentic-Ai/src/workflows/langgraph_phase5.py → plan_edit_node.
"""

import sys
from pathlib import Path

_AGENTIC_ROOT = Path(__file__).resolve().parents[2] / "Agentic-Ai"
if str(_AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTIC_ROOT))

from src.workflows.langgraph_phase5 import plan_edit_node  # noqa: E402, F401

__all__ = ["plan_edit_node"]
