"""
edit_agent/agent.py  –  Re-export wrapper for Phase 5 Edit Agent

The actual implementation lives in Agentic-Ai/src/workflows/langgraph_phase5.py.
This module provides a convenient public interface from the outer directory.
"""

import sys
from pathlib import Path

# Ensure the Agentic-Ai package root is importable
_AGENTIC_ROOT = Path(__file__).resolve().parents[2] / "Agentic-Ai"
if str(_AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTIC_ROOT))

from src.workflows.langgraph_phase5 import (  # noqa: E402, F401
    build_graph,
    run_edit,
    Phase5State,
)

__all__ = ["build_graph", "run_edit", "Phase5State"]
