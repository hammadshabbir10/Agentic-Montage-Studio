"""
state_manager/history.py  –  Re-export wrapper

The history functionality is implemented in StateManager.history().
See Agentic-Ai/src/state_versioning.py.
"""

import sys
from pathlib import Path

_AGENTIC_ROOT = Path(__file__).resolve().parents[1] / "Agentic-Ai"
if str(_AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTIC_ROOT))

from src.state_versioning import StateManager  # noqa: E402, F401

# Convenience alias
history = StateManager.history

__all__ = ["StateManager", "history"]
