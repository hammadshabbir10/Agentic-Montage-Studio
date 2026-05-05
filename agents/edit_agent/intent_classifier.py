"""
edit_agent/intent_classifier.py  –  Re-export wrapper for Phase 5 Intent Classifier

The actual implementation lives in Agentic-Ai/src/agents/edit_intent_classifier.py.
"""

import sys
from pathlib import Path

_AGENTIC_ROOT = Path(__file__).resolve().parents[2] / "Agentic-Ai"
if str(_AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTIC_ROOT))

from src.agents.edit_intent_classifier import (  # noqa: E402, F401
    EditIntent,
    classify,
    classify_without_llm,
)

__all__ = ["EditIntent", "classify", "classify_without_llm"]
