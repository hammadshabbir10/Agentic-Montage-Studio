"""
state_manager/snapshot.py  –  Re-export wrapper

The snapshot functionality is implemented in StateManager.snapshot().
See Agentic-Ai/src/state_versioning.py.
"""

import sys
from pathlib import Path

_AGENTIC_ROOT = Path(__file__).resolve().parents[1] / "Agentic-Ai"
if str(_AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTIC_ROOT))

from src.state_versioning import StateManager  # noqa: E402, F401

# Convenience alias
snapshot = StateManager.snapshot

__all__ = ["StateManager", "snapshot"]
