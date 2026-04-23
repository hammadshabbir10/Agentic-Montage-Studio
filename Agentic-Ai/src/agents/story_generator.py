"""
story_generator.py

Generates a structured story JSON from either:
  - A free-form user prompt (auto mode)
  - An existing script text (manual mode)

Output schema (consumed by build_story_manifest):
  {
    "title":       str,
    "logline":     str,          # one-sentence pitch
    "genre":       str,
    "tone":        str,
    "setting":     str,
    "time_period": str,
    "themes":      [str, ...],
    "acts": [
      {
        "act":         int,       # 1 = intro, 2 = conflict, 3 = climax, 4 = resolution
        "label":       str,       # e.g. "Introduction", "Rising Conflict"
        "description": str        # 2-3 sentences
      },
      ...
    ],
    "protagonist": str,
    "antagonist":  str | null,
    "world":       str            # brief world-building note
  }
"""

import json
import re
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _prompt_from_user_input(user_prompt: str, num_scenes: int) -> str:
    return (
        "You are a professional story architect for film and television.\n\n"
        f"User prompt: {user_prompt}\n"
        f"Number of scenes: {num_scenes}\n\n"
        "Based ONLY on the prompt above, write a JSON object (no markdown, no extra text) "
        "with EXACTLY these keys:\n"
        '  "title":       a short, evocative title for the story,\n'
        '  "logline":     a single sentence that captures the core conflict and stakes,\n'
        '  "genre":       primary genre (e.g. "Cold War Thriller", "Sci-Fi Drama"),\n'
        '  "tone":        overall tone (e.g. "tense and gritty", "whimsical and hopeful"),\n'
        '  "setting":     primary physical setting,\n'
        '  "time_period": era or time period (e.g. "1970s Cold War Berlin"),\n'
        '  "themes":      a JSON array of 2-4 thematic keywords,\n'
        '  "acts": a JSON array of 4 objects, each with keys "act" (int 1-4), '
        '"label" (e.g. "Introduction"), and "description" (2-3 sentences describing '
        'that narrative beat — intro, conflict, climax, resolution),\n'
        '  "protagonist": name or brief description of the main character,\n'
        '  "antagonist":  name or brief description of the main opposing force (or null),\n'
        '  "world":       one sentence of world-building context.\n'
        "Return ONLY valid JSON."
    )


def _prompt_from_script(script_text: str) -> str:
    # Feed only the first ~1500 chars to keep the prompt manageable
    excerpt = script_text[:1500].strip()
    return (
        "You are a professional story architect for film and television.\n\n"
        "The following is the opening of a screenplay:\n"
        f"---\n{excerpt}\n---\n\n"
        "Infer the story structure from the script above and write a JSON object "
        "(no markdown, no extra text) with EXACTLY these keys:\n"
        '  "title":       a short title inferred from the script,\n'
        '  "logline":     a single sentence that captures the core conflict and stakes,\n'
        '  "genre":       primary genre,\n'
        '  "tone":        overall tone,\n'
        '  "setting":     primary physical setting,\n'
        '  "time_period": era or time period,\n'
        '  "themes":      a JSON array of 2-4 thematic keywords,\n'
        '  "acts": a JSON array of 4 objects, each with keys "act" (int 1-4), '
        '"label", and "description" (2-3 sentences),\n'
        '  "protagonist": name or brief description of the main character,\n'
        '  "antagonist":  name or brief description of the main opposing force (or null),\n'
        '  "world":       one sentence of world-building context.\n'
        "Return ONLY valid JSON."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_story_response(text: str) -> Dict[str, Any]:
    """Robustly extract the JSON object from the LLM response."""
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback defaults so the pipeline never crashes
    return {
        "title": "Untitled Story",
        "logline": "Story details could not be parsed.",
        "genre": "Drama",
        "tone": "Dramatic",
        "setting": "Unknown",
        "time_period": "Contemporary",
        "themes": ["conflict", "survival"],
        "acts": [
            {"act": 1, "label": "Introduction",      "description": "The story begins."},
            {"act": 2, "label": "Rising Conflict",   "description": "Tension escalates."},
            {"act": 3, "label": "Climax",            "description": "The crisis peaks."},
            {"act": 4, "label": "Resolution",        "description": "The conflict resolves."},
        ],
        "protagonist": "Unknown",
        "antagonist": None,
        "world": "Details unavailable.",
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_from_prompt(
    prompt: str,
    num_scenes: int,
    tool_client,
) -> Dict[str, Any]:
    """Generate a story manifest from a free-form user prompt (auto mode)."""
    llm_prompt = _prompt_from_user_input(prompt, num_scenes)
    try:
        result = tool_client.invoke_by_capability(
            "generate_script_segment", {"prompt": llm_prompt}
        )
        story = _parse_story_response(result.get("text", ""))
    except Exception:
        story = _parse_story_response("")
    return story


def run_from_script(
    script_text: str,
    tool_client,
) -> Dict[str, Any]:
    """Infer a story manifest from an existing screenplay (manual mode)."""
    llm_prompt = _prompt_from_script(script_text)
    try:
        result = tool_client.invoke_by_capability(
            "generate_script_segment", {"prompt": llm_prompt}
        )
        story = _parse_story_response(result.get("text", ""))
    except Exception:
        story = _parse_story_response("")
    return story