"""
character_designer.py

Builds a rich character profile for every unique character extracted from
the scene manifest.  Each character gets:
  - personality  (derived from their dialogue samples)
  - appearance   (era-appropriate, role-specific)
  - role         (protagonist / antagonist / supporting / etc.)
  - style_reference (visual/cinematic style matching the story)
  - first_appearance (scene_id)
  - dialogue_samples (list of {line, visual_cue})
"""

from typing import Any, Dict, List
import json
import re


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_characters(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Walk all scenes and build a per-character map:
      { "AUGIE": { first_appearance, dialogue_samples: [...] } }
    Character names are already normalised (no ** etc.) by script_ingest.
    """
    char_map: Dict[str, Dict[str, Any]] = {}

    for scene in manifest.get("scenes", []):
        scene_id = scene.get("scene_id", 0)
        for d in scene.get("dialogue", []):
            name = d.get("speaker", "").strip()
            if not name:
                continue
            if name not in char_map:
                char_map[name] = {
                    "first_appearance": scene_id,
                    "dialogue_samples": [],
                }
            char_map[name]["dialogue_samples"].append({
                "line": d.get("line", ""),
                "visual_cue": d.get("visual_cue", ""),
            })

    return char_map


def _build_profile_prompt(
    name: str,
    dialogue_samples: List[Dict[str, str]],
    story_context: str,
) -> str:
    samples_text = "\n".join(
        f'  - "{s["line"]}"' for s in dialogue_samples[:4]
    )
    return (
        f"You are a character designer for a film production.\n"
        f"Story context: {story_context}\n\n"
        f"Character name: {name}\n"
        f"Sample dialogue lines:\n{samples_text}\n\n"
        "Based ONLY on the story context and dialogue above, write a short JSON object "
        "(no markdown, no extra text) with exactly these keys:\n"
        '  "personality": a 1-2 sentence description of this character\'s personality '
        "and motivations, inferred from their dialogue,\n"
        '  "appearance": a 1-2 sentence physical description appropriate to the era and '
        "role of this character,\n"
        '  "role": one of [protagonist, antagonist, supporting, minor],\n'
        '  "style_reference": a short cinematic style note (e.g. "Cold War noir, muted '
        "tones, trench coat aesthetic\").\n"
        "Return ONLY valid JSON."
    )


def _normalize_field(value: Any) -> str:
    """Normalize a field that might be a string or dict to a string."""
    if isinstance(value, dict):
        # If it's a dict, try to get the 'description' key, otherwise join all string values
        if "description" in value:
            return str(value["description"])
        else:
            return " ".join(str(v) for v in value.values() if isinstance(v, str))
    else:
        return str(value)


def _parse_profile_response(text: str) -> Dict[str, str]:
    """Extract the JSON object from the LLM response robustly."""
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback — return safe defaults so the pipeline doesn't crash
    return {
        "personality": "Character details could not be parsed.",
        "appearance": "Appearance details could not be parsed.",
        "role": "supporting",
        "style_reference": "Cinematic",
    }


def _infer_story_context(manifest: Dict[str, Any]) -> str:
    """
    Build a brief context string from scene locations so the LLM
    knows what kind of story it is designing for.
    """
    locations = list(dict.fromkeys(
        s.get("location", "") for s in manifest.get("scenes", []) if s.get("location")
    ))
    return f"Locations include: {', '.join(locations[:6])}."


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(manifest: Dict[str, Any], tool_client) -> List[Dict[str, Any]]:
    """
    Returns a list of character dicts ready to be passed to build_character_db().
    Each character has unique personality, appearance, role, and style_reference.
    """
    char_map = _collect_characters(manifest)
    story_context = _infer_story_context(manifest)
    characters: List[Dict[str, Any]] = []

    for name, data in char_map.items():
        dialogue_samples = data["dialogue_samples"]

        # Ask the LLM for a unique profile
        prompt = _build_profile_prompt(name, dialogue_samples, story_context)
        try:
            result = tool_client.invoke_by_capability(
                "generate_script_segment", {"prompt": prompt}
            )
            response_text = result.get("text", "")
            profile = _parse_profile_response(response_text)
        except Exception:
            profile = {
                "personality": f"{name} is a determined character whose motives are not yet clear.",
                "appearance": "Period-appropriate attire, distinctive features.",
                "role": "supporting",
                "style_reference": "Cinematic",
            }

        characters.append({
            "name": name,
            "personality": _normalize_field(profile.get("personality", "Unknown")),
            "appearance": _normalize_field(profile.get("appearance", "Unknown")),
            "role": profile.get("role", "supporting"),
            "style_reference": profile.get("style_reference", "Cinematic"),
            "first_appearance": data["first_appearance"],
            "dialogue_samples": dialogue_samples,
        })

    return characters