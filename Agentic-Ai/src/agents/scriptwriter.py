import re
from typing import Dict
from src.io.script_ingest import parse_script_to_manifest, _normalize_character_name


def _clean_line(line: str) -> str:
    """Remove leading parentheticals like (smiling) from dialogue."""
    return re.sub(r"^\([^)]*\)\s*", "", line).strip()


def _deduplicate_characters(scenes):
    """
    Ensure character names are normalized (no trailing **) and
    deduplicated within each scene's characters list.
    """
    for scene in scenes:
        # Normalize speaker names in dialogue
        for d in scene.get("dialogue", []):
            d["speaker"] = _normalize_character_name(d["speaker"])
        # Rebuild characters list from actual dialogue speakers
        seen = []
        for d in scene.get("dialogue", []):
            name = d["speaker"]
            if name not in seen:
                seen.append(name)
        scene["characters"] = seen
    return scenes


def _fill_missing_visual_cues(scenes):
    """
    For each character in a scene, if any dialogue line has a real visual cue,
    propagate it to lines that still have the generic fallback. Also ensure
    no line ever has an empty cue.
    """
    for scene in scenes:
        # Collect best known cue per character
        char_best_cue: Dict[str, str] = {}
        for d in scene.get("dialogue", []):
            speaker = d["speaker"]
            cue = d.get("visual_cue", "")
            if cue and cue != "Default visual cue.":
                char_best_cue[speaker] = cue

        for d in scene.get("dialogue", []):
            speaker = d["speaker"]
            cue = d.get("visual_cue", "")
            if not cue or cue == "Default visual cue.":
                if speaker in char_best_cue:
                    d["visual_cue"] = char_best_cue[speaker]
                else:
                    # Generate a meaningful fallback
                    location = scene.get("location", "scene")
                    d["visual_cue"] = (
                        f"Medium shot of {speaker} at {location}, "
                        "expression intense and focused."
                    )
    return scenes


def run(prompt: str, tool_client) -> Dict[str, object]:
    from src.io.json_schema import build_scene_manifest

    result = tool_client.invoke_by_capability(
        "generate_script_segment", {"prompt": prompt}
    )
    script_text = result.get("text", "")
    manifest_raw = parse_script_to_manifest(script_text, title="Autonomous Script")
    scenes = manifest_raw.get("scenes", [])

    # Clean up dialogue lines
    for scene in scenes:
        for d in scene.get("dialogue", []):
            d["line"] = _clean_line(d["line"])

    # Fix character names and cues
    scenes = _deduplicate_characters(scenes)
    scenes = _fill_missing_visual_cues(scenes)

    manifest = build_scene_manifest(scenes)
    return {"script_text": script_text, "manifest": manifest}