from typing import Any, Dict, List
import datetime


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------
# Average spoken English: ~130 words/min = ~2.2 words/second
_WORDS_PER_SECOND = 2.2
# Pause between speakers (breath, beat)
_LINE_PAUSE_SECONDS = 0.8
# Minimum scene duration even with no dialogue
_MIN_SCENE_SECONDS = 4
# Extra seconds for establishing shot / scene atmosphere
_SCENE_INTRO_SECONDS = 2


def _estimate_line_duration(line: str) -> float:
    """Seconds to speak one dialogue line."""
    clean = line.strip().strip('"').strip("'").strip()
    word_count = len(clean.split()) if clean else 0
    if word_count == 0:
        return 0.0
    return (word_count / _WORDS_PER_SECOND) + _LINE_PAUSE_SECONDS


def _estimate_scene_duration(scene: Dict[str, Any]) -> int:
    """
    Total scene duration in whole seconds.
    = INTRO_PADDING + sum of each dialogue line's speaking time
    Minimum: MIN_SCENE_SECONDS
    """
    total = float(_SCENE_INTRO_SECONDS)
    for d in scene.get("dialogue", []):
        total += _estimate_line_duration(d.get("line", ""))
    return max(_MIN_SCENE_SECONDS, round(total))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_scene_manifest(scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
    workflow_id = f"workflow_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.datetime.now().isoformat()

    total_duration = 0

    for scene in scenes:
        # Always recalculate — never trust a hardcoded 10
        scene["duration"] = _estimate_scene_duration(scene)
        total_duration += scene["duration"]

        # Strip internal action lines from output
        scene.pop("actions", None)

        # Guarantee every dialogue line has a real visual cue
        for d in scene.get("dialogue", []):
            cue = d.get("visual_cue", "").strip()
            if not cue or cue == "Default visual cue.":
                speaker = d.get("speaker", "CHARACTER")
                location = scene.get("location", "scene")
                d["visual_cue"] = (
                    f"Medium shot of {speaker} at {location}, expression intense."
                )

    return {
        "workflow_id": workflow_id,
        "timestamp": timestamp,
        "scenes": scenes,
        "total_scenes": len(scenes),
        "total_duration_seconds": total_duration,
    }


def build_character_db(characters: List[Dict[str, Any]]) -> Dict[str, Any]:
    workflow_id = f"workflow_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.datetime.now().isoformat()

    return {
        "workflow_id": workflow_id,
        "timestamp": timestamp,
        "characters": characters,
        "total_characters": len(characters),
    }


def build_story_manifest(story: Dict[str, Any]) -> Dict[str, Any]:
    workflow_id = f"workflow_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.datetime.now().isoformat()

    safe_story = {
        "title":       story.get("title", "Untitled"),
        "logline":     story.get("logline", ""),
        "genre":       story.get("genre", "Drama"),
        "tone":        story.get("tone", "Dramatic"),
        "setting":     story.get("setting", ""),
        "time_period": story.get("time_period", ""),
        "themes":      story.get("themes", []),
        "acts":        story.get("acts", []),
        "protagonist": story.get("protagonist", ""),
        "antagonist":  story.get("antagonist", None),
        "world":       story.get("world", ""),
    }

    return {
        "workflow_id": workflow_id,
        "timestamp":   timestamp,
        "story":       safe_story,
    }