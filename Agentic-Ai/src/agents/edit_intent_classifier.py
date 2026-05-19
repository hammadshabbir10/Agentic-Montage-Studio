"""
edit_intent_classifier.py  –  Phase 5 LLM-Powered Edit Intent Classification

Uses Groq LLM (same as Phase 1) to parse free-text edit queries into
structured EditIntent objects per PDF Section 5.2.

Covers 10+ edit query types:
  - change_voice_tone        → audio
  - make_scene_darker        → video_frame
  - add_background_music     → audio
  - remove_subtitle          → video
  - change_character_design  → video_frame
  - speed_up_scene           → video
  - regenerate_script        → script
  - apply_filter             → video_frame
  - change_scene_mood        → audio
  - adjust_volume            → audio
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ── Pydantic schema for classified intent ────────────────────────────────────

class EditIntent(BaseModel):
    """Structured intent object returned by the classifier."""
    intent: str = Field(
        ...,
        description="Canonical edit action, e.g. 'change_voice_tone', 'apply_filter'",
    )
    target: str = Field(
        ...,
        description="Target component: 'audio' | 'video_frame' | 'video' | 'script'",
    )
    scope: str = Field(
        default="all",
        description="Scope of the edit, e.g. 'scene:1', 'character:Narrator', 'all'",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the edit, e.g. {'tone': 'whispered', 'filter': 'sepia'}",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Classifier confidence score 0.0–1.0",
    )


# ── System prompt for intent classification ──────────────────────────────────

_SYSTEM_PROMPT = """\
You are an intelligent edit-intent classifier for an AI video generation pipeline.

The pipeline has four target components that can be edited:
1. **audio** – Voice synthesis (TTS) and background music
2. **video_frame** – Still image generation for scenes (including visual filters)
3. **video** – Full video composition/export (speed, subtitles, transitions)
4. **script** – Story and screenplay content (re-invokes the LLM)

Given a user's free-text edit command, classify it into a structured JSON object:

{
  "intent": "<canonical_action>",
  "target": "<audio|video_frame|video|script>",
  "scope": "<scene:N|character:NAME|all>",
  "parameters": { ... relevant parameters ... },
  "confidence": <0.0 to 1.0>
}

## Known intent types and their targets:

| Intent | Target | Example Parameters |
|--------|--------|--------------------|
| change_voice_tone | audio | {"tone": "whispered", "character": "Narrator"} |
| add_background_music | audio | {"mood": "tense", "scene_id": 1} |
| adjust_volume | audio | {"adjustment": "louder", "factor": 1.5} |
| make_scene_darker | video_frame | {"brightness": -0.3, "scene_id": 1} |
| make_scene_brighter | video_frame | {"brightness": 0.3, "scene_id": 1} |
| apply_filter | video_frame | {"filter": "sepia", "scene_id": 1} |
| change_character_design | video_frame | {"character": "Jack", "description": "wearing a hat"} |
| remove_subtitle | video | {"subtitles": false} |
| speed_up_scene | video | {"speed_factor": 1.5, "scene_id": 2} |
| slow_down_scene | video | {"speed_factor": 0.5, "scene_id": 2} |
| regenerate_script | script | {"prompt_modifier": "make it funnier"} |
| change_scene_mood | audio | {"mood": "happy", "scene_id": 1} |

## Rules:
- Extract scene numbers from the query when mentioned (e.g. "scene 2" → scope: "scene:2")
- Extract character names when mentioned (e.g. "Jack's voice" → scope: "character:Jack")
- For filter requests, identify the filter type: sepia, blur, sharpen, grayscale, vignette, brightness, contrast
- If the query is ambiguous, set confidence < 0.7
- If you cannot classify, use intent "unknown" with target "video" and confidence 0.1
- Return ONLY the JSON object, no explanation.
"""


# ── Classifier ───────────────────────────────────────────────────────────────

def classify(
    query: str,
    tool_client: Any,
    context: Optional[Dict[str, Any]] = None,
) -> EditIntent:
    """
    Classify a free-text edit query into a structured EditIntent.

    Parameters
    ----------
    query : str
        The user's raw edit command (e.g. "Make scene 1 darker").
    tool_client : ToolClient
        The MCP tool client for invoking Groq LLM.
    context : dict, optional
        Optional pipeline context (current scenes, characters, etc.) to
        help the classifier make better decisions.

    Returns
    -------
    EditIntent
        Parsed and validated intent object.
    """
    # First try the fast rule-based path for common patterns
    rule_result = _rule_based_classify(query)
    if rule_result and rule_result.confidence >= 0.9:
        return rule_result

    # Fall back to LLM classification
    context_hint = ""
    if context:
        scenes = context.get("scenes", [])
        characters = context.get("characters", [])
        if scenes:
            context_hint += f"\nAvailable scenes: {[s.get('scene_id') for s in scenes]}"
        if characters:
            context_hint += f"\nAvailable characters: {[c.get('name') for c in characters]}"

    user_msg = f"Edit command: {query}{context_hint}"

    try:
        result = tool_client.invoke_by_capability(
            "classify_edit_intent",
            {"prompt": user_msg},
        )
        raw_text = result.get("text", "")
        intent_dict = _parse_llm_json(raw_text)
        return EditIntent(**intent_dict)
    except Exception:
        # If LLM fails, fall back to rule-based
        if rule_result:
            return rule_result
        return _fallback_classify(query)


def classify_without_llm(query: str) -> EditIntent:
    """
    Classify using only rule-based matching (no LLM call).
    Useful for testing and when Groq API is unavailable.
    """
    result = _rule_based_classify(query)
    if result:
        return result
    return _fallback_classify(query)


# ── Rule-based fast path ─────────────────────────────────────────────────────

_RULES = [
    # (pattern, intent, target, param_builder)
    (r"(?:change|modify|alter)\s+(?:the\s+)?voice\s*(?:tone)?",
     "change_voice_tone", "audio",
     lambda m, q: {"tone": _extract_tone(q)}),

    (r"(?:make|set)\s+(?:the\s+)?(?:scene\s*\d*\s+)?(?:darker|dim)",
     "make_scene_darker", "video_frame",
     lambda m, q: {"brightness": -0.3, "scene_id": _extract_scene_id(q)}),

    (r"(?:make|set)\s+(?:the\s+)?(?:scene\s*\d*\s+)?(?:brighter|lighter)",
     "make_scene_brighter", "video_frame",
     lambda m, q: {"brightness": 0.3, "scene_id": _extract_scene_id(q)}),

    (r"(?:add|include|insert)\s+(?:a\s+)?(?:background\s+)?music",
     "add_background_music", "audio",
     lambda m, q: {"mood": _extract_mood(q), "scene_id": _extract_scene_id(q)}),

    (r"(?:remove|disable|turn\s+off|hide)\s+(?:the\s+)?subtitle",
     "remove_subtitle", "video",
     lambda m, q: {"subtitles": False}),

    (r"(?:change|modify|redesign|update)\s+(?:the\s+)?character\s*(?:design)?",
     "change_character_design", "video_frame",
     lambda m, q: {
         "character": _extract_character(q),
         "description": _extract_character_description(q),
     }),

    (r"(?:speed\s+up|faster|accelerate)\s+(?:the\s+)?(?:scene|this)",
     "speed_up_scene", "video",
     lambda m, q: {"speed_factor": 1.5, "scene_id": _extract_scene_id(q)}),

    (r"(?:slow\s+down|slower|decelerate)\s+(?:the\s+)?(?:scene|this)",
     "slow_down_scene", "video",
     lambda m, q: {"speed_factor": 0.5, "scene_id": _extract_scene_id(q)}),

    (r"(?:regenerate|rewrite|redo)\s+(?:the\s+)?(?:script|story|screenplay)",
     "regenerate_script", "script",
     lambda m, q: {"prompt_modifier": q}),

    (r"(?:apply|add|use)\s+(?:a\s+)?(?:sepia|blur|sharpen|grayscale|vignette|brightness|contrast)\s*(?:filter)?",
     "apply_filter", "video_frame",
     lambda m, q: {"filter": _extract_filter(q), "scene_id": _extract_scene_id(q)}),

    (r"(?:change|set|modify)\s+(?:the\s+)?(?:scene\s*\d*\s+)?mood",
     "change_scene_mood", "audio",
     lambda m, q: {"mood": _extract_mood(q), "scene_id": _extract_scene_id(q)}),

    (r"(?:adjust|change|increase|decrease|raise|lower)\s+(?:the\s+)?volume",
     "adjust_volume", "audio",
     lambda m, q: {"adjustment": _extract_volume_adj(q), "scene_id": _extract_scene_id(q)}),

    # Additional filter patterns
    (r"(?:make|turn)\s+(?:the\s+)?(?:scene\s*\d*\s+)?(?:black\s*(?:and|&)\s*white|grayscale|grey)",
     "apply_filter", "video_frame",
     lambda m, q: {"filter": "grayscale", "scene_id": _extract_scene_id(q)}),

    (r"(?:make|turn)\s+(?:the\s+)?(?:scene\s*\d*\s+)?(?:sepia|vintage|old)",
     "apply_filter", "video_frame",
     lambda m, q: {"filter": "sepia", "scene_id": _extract_scene_id(q)}),

    (r"(?:blur|soften)\s+(?:the\s+)?(?:scene|image|frame)",
     "apply_filter", "video_frame",
     lambda m, q: {"filter": "blur", "scene_id": _extract_scene_id(q)}),

    (r"(?:sharpen|crisp)\s+(?:the\s+)?(?:scene|image|frame)",
     "apply_filter", "video_frame",
     lambda m, q: {"filter": "sharpen", "scene_id": _extract_scene_id(q)}),

    # Visual prompt modification: "Change the blue car to red in scene 2", "Make the sky green in scene 1"
    (r"(?:change|make|turn|set)\s+(?:the\s+)?(.*?)\s+(?:to|into|as|changed\s+into|to\s+be)\s+(.*?)\s+in\s+scene\s*(\d+)",
     "modify_scene_visuals", "modify_scene_visuals",
     lambda m, q: {
         "original": re.sub(r"^(?:a|an|the)\s+", "", m.group(1).strip(), flags=re.IGNORECASE),
         "replacement": m.group(2).strip(),
         "scene_id": int(m.group(3)),
     }),
]


def _rule_based_classify(query: str) -> Optional[EditIntent]:
    """Try to match the query against known patterns."""
    q_lower = query.lower().strip()
    for pattern, intent, target, param_builder in _RULES:
        match = re.search(pattern, q_lower)
        if match:
            params = param_builder(match, query) if param_builder else {}
            scope = _extract_scope(query)
            return EditIntent(
                intent=intent,
                target=target,
                scope=scope,
                parameters=params,
                confidence=0.95,
            )
    return None


def _fallback_classify(query: str) -> EditIntent:
    """Last-resort classification when nothing else works."""
    return EditIntent(
        intent="unknown",
        target="video",
        scope="all",
        parameters={"raw_query": query},
        confidence=0.1,
    )


# ── Extraction helpers ───────────────────────────────────────────────────────

def _extract_scene_id(query: str) -> Optional[int]:
    match = re.search(r"scene\s*(\d+)", query, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_character(query: str) -> Optional[str]:
    # Look for quoted names first
    match = re.search(r'["\']([^"\' ]+)["\']', query)
    if match:
        return match.group(1).strip()
    # Names after the word "character"
    match = re.search(r"character\s+(?:design\s+(?:of|for)\s+)?([A-Z][a-zA-Z]+)", query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # ALL-CAPS or Title-Case name after make/give/set/change/update, stops before 'a'/'an'
    match = re.search(
        r"(?:make|give|set|change|update)\s+([A-Z][A-Z0-9]+(?:\s+[A-Z][A-Z0-9]+)*)\s+(?:a|an)\b",
        query,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def _extract_character_name_from_query(query: str) -> Optional[str]:
    """Extract ALL-CAPS or Title-Case character name from queries like 'Make OWAI a black skin'."""
    # Capture NAME that comes right before 'a'/'an' + trait
    match = re.search(
        r"(?:make|give|set|change|update)\s+([A-Z][A-Z0-9]+(?:\s+[A-Z][A-Z0-9]+)*)\s+(?:a|an)\b",
        query,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return _extract_character(query)


def _extract_appearance_description(query: str) -> Optional[str]:
    """Extract appearance trait from queries like 'Make OWAI a black skin color'."""
    # Match: make/give/set/change NAME [a/an/the] <trait> [in scene N]
    # Uses \S+ for the character name token to avoid space-in-class ambiguity
    match = re.search(
        r"(?:make|give|set|change|update)\s+\S+\s+(?:a|an|to have|the)?\s*(.+?)(?:\s+in\s+scene\s*\d+)?$",
        query,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().strip(".")
    return _extract_character_description(query)


def _extract_character_description(query: str) -> Optional[str]:
    # Try to capture the design change after "to/with/as"
    match = re.search(
        r"(?:design|character)(?:\s+of\s+|\s+for\s+)?[^.]*?\s+(?:to|with|as)\s+(.+)$",
        query,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().strip(".")
    match = re.search(
        r"(?:change|modify|redesign|update)\s+[^.]*?\s+(?:to|with|as)\s+(.+)$",
        query,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().strip(".")
    return None


def _extract_tone(query: str) -> str:
    tones = ["whispered", "loud", "soft", "deep", "high", "angry", "calm",
             "happy", "sad", "excited", "serious", "dramatic"]
    q_lower = query.lower()
    for tone in tones:
        if tone in q_lower:
            return tone
    return "default"


def _extract_mood(query: str) -> str:
    moods = ["tense", "happy", "sad", "mysterious", "action", "hopeful",
             "dark", "romantic", "epic", "calm", "dramatic", "suspenseful"]
    q_lower = query.lower()
    for mood in moods:
        if mood in q_lower:
            return mood
    return "neutral"


def _extract_filter(query: str) -> str:
    filters = ["sepia", "blur", "sharpen", "grayscale", "vignette",
               "brightness", "contrast"]
    q_lower = query.lower()
    for f in filters:
        if f in q_lower:
            return f
    return "sepia"


def _extract_volume_adj(query: str) -> str:
    q_lower = query.lower()
    if any(w in q_lower for w in ["increase", "raise", "louder", "up"]):
        return "louder"
    if any(w in q_lower for w in ["decrease", "lower", "quieter", "down"]):
        return "quieter"
    return "louder"


def _extract_scope(query: str) -> str:
    scene_id = _extract_scene_id(query)
    if scene_id is not None:
        return f"scene:{scene_id}"
    character = _extract_character(query)
    if character:
        return f"character:{character}"
    return "all"


def _parse_llm_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from LLM response text."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding { ... } in the text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"intent": "unknown", "target": "video", "confidence": 0.1}
