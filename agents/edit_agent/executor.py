"""
edit_agent/executor.py  –  Re-export wrapper for Phase 5 Edit Executor

The actual implementation lives in Agentic-Ai/src/agents/edit_executor.py.
"""

import sys
from pathlib import Path

_AGENTIC_ROOT = Path(__file__).resolve().parents[2] / "Agentic-Ai"
if str(_AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTIC_ROOT))

from src.agents.edit_executor import (  # noqa: E402, F401
    execute,
    collect_current_state,
    collect_current_asset_paths,
)

__all__ = ["execute", "collect_current_state", "collect_current_asset_paths"]
