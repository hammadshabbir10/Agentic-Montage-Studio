import re
from typing import Any, Dict, List

# Standard screenplay scene headings only — NOT bare numbers like "1:"
SCENE_HEADING_RE = re.compile(
    r"^(INT\.|EXT\.|INT/EXT\.|I/E\.)\s+\S+", re.IGNORECASE
)
VISUAL_CUE_RE = re.compile(r"^(VISUAL\s*CUE|VISUAL|SHOT|CUE)\s*:", re.IGNORECASE)
TRANSITION_RE = re.compile(
    r"^(FADE\s*IN|FADE\s*OUT|FADE\s*TO\s*BLACK|CUT\s*TO)(\.|:)?$", re.IGNORECASE
)


def _normalize_character_name(name: str) -> str:
    """Strip trailing ** and other markdown artifacts, uppercase."""
    cleaned = re.sub(r"[\*_`]+$", "", name.strip())
    cleaned = re.sub(r"^[\*_`]+", "", cleaned)
    return cleaned.strip().upper()


def validate_script_structure(script_text: str) -> List[str]:
    errors: List[str] = []
    lines = [_normalize_line(line) for line in script_text.splitlines()]
    lines = [line for line in lines if line]
    has_scene = any(SCENE_HEADING_RE.match(line) for line in lines)
    has_dialogue = any(_is_dialogue_line(line) for line in lines)
    has_action = any(_is_action_line(line) for line in lines)
    if not has_scene:
        errors.append("Missing scene headings (INT./EXT.)")
    if not has_dialogue:
        errors.append("Missing dialogue labels (e.g., NAME: line)")
    if not has_action:
        errors.append("Missing action descriptions")
    return errors


def parse_script_to_manifest(script_text: str, title: str = "Untitled") -> Dict[str, Any]:
    scenes: List[Dict[str, Any]] = []
    current_scene: Dict[str, Any] = {}
    action_lines: List[str] = []
    dialogue: List[Dict[str, str]] = []
    current_visual_cue = ""
    pending_speaker = ""
    pending_parenthetical = ""
    characters: List[str] = []

    def flush_scene() -> None:
        if not current_scene:
            return
        # Only keep scenes that have dialogue or meaningful actions
        if not dialogue and not action_lines:
            return
        if action_lines:
            current_scene["actions"] = list(action_lines)
        current_scene["dialogue"] = list(dialogue)
        current_scene["characters"] = list(dict.fromkeys(characters))
        scenes.append(dict(current_scene))

    for raw_line in script_text.splitlines():
        raw = _normalize_line(raw_line)
        if not raw:
            continue

        if TRANSITION_RE.match(raw):
            continue

        # Scene heading
        if SCENE_HEADING_RE.match(raw):
            flush_scene()
            current_scene = {
                "scene_id": len(scenes) + 1,
                "location": _extract_location(raw),
            }
            action_lines = []
            dialogue = []
            current_visual_cue = ""
            characters = []
            pending_speaker = ""
            pending_parenthetical = ""
            continue

        # Visual cue line
        if VISUAL_CUE_RE.match(raw):
            current_visual_cue = raw.split(":", 1)[1].strip() if ":" in raw else raw
            continue

        # Inline dialogue: CHARACTER NAME: "line"
        if _is_dialogue_line(raw):
            speaker_raw, line_text = raw.split(":", 1)
            speaker_name = _normalize_character_name(speaker_raw)
            if TRANSITION_RE.match(speaker_name):
                continue
            visual = current_visual_cue if current_visual_cue else _default_visual_cue(speaker_name)
            dialogue.append({
                "speaker": speaker_name,
                "line": line_text.strip(),
                "visual_cue": visual,
            })
            if speaker_name not in characters:
                characters.append(speaker_name)
            current_visual_cue = ""
            continue

        # Standalone speaker name (screenplay block format)
        if _is_speaker_line(raw):
            pending_speaker = _normalize_character_name(raw)
            pending_parenthetical = ""
            continue

        # Parenthetical after speaker
        if pending_speaker and raw.startswith("(") and raw.endswith(")"):
            pending_parenthetical = raw
            continue

        # Dialogue line following a standalone speaker
        if pending_speaker:
            line_text = raw
            if pending_parenthetical:
                line_text = f"{pending_parenthetical} {line_text}".strip()
            visual = current_visual_cue if current_visual_cue else _default_visual_cue(pending_speaker)
            dialogue.append({
                "speaker": pending_speaker,
                "line": line_text.strip(),
                "visual_cue": visual,
            })
            if pending_speaker not in characters:
                characters.append(pending_speaker)
            pending_speaker = ""
            pending_parenthetical = ""
            current_visual_cue = ""
            continue

        # Action line
        if _is_action_line(raw):
            action_lines.append(raw)

    # Flush any pending speaker at EOF
    if pending_speaker:
        dialogue.append({
            "speaker": pending_speaker,
            "line": pending_parenthetical.strip(),
            "visual_cue": current_visual_cue if current_visual_cue else _default_visual_cue(pending_speaker),
        })
        if pending_speaker not in characters:
            characters.append(pending_speaker)

    flush_scene()

    # Fallback: no scene headings parsed but there is content
    if not scenes and (action_lines or dialogue):
        fallback_scene = {
            "scene_id": 1,
            "location": "UNKNOWN",
            "dialogue": list(dialogue),
            "characters": list(dict.fromkeys(characters)),
        }
        if action_lines:
            fallback_scene["actions"] = list(action_lines)
        scenes.append(fallback_scene)

    # Re-number scene IDs after filtering
    for idx, scene in enumerate(scenes, start=1):
        scene["scene_id"] = idx

    return {"scenes": scenes}


def _default_visual_cue(speaker: str) -> str:
    """Generate a generic but non-empty visual cue when none is provided."""
    return f"Medium shot of {speaker}, expression intense and focused."


def _is_valid_speaker(name: str) -> bool:
    """
    A valid speaker name is ALL CAPS, max 40 chars, and max 4 words.
    This prevents action lines like 'THE MOON CASTS AN EERIE GLOW...'
    from being mistaken for character names.
    """
    if not name:
        return False
    if len(name) > 40:
        return False
    words = name.split()
    if len(words) > 4:
        return False
    return name.isupper()


def _is_dialogue_line(line: str) -> bool:
    if ":" not in line:
        return False
    speaker = line.split(":", 1)[0].strip()
    speaker_clean = _normalize_character_name(speaker)
    if not speaker_clean:
        return False
    if TRANSITION_RE.match(speaker_clean):
        return False
    if SCENE_HEADING_RE.match(line):
        return False
    if VISUAL_CUE_RE.match(line):
        return False
    return _is_valid_speaker(speaker_clean)


def _is_speaker_line(line: str) -> bool:
    if not line:
        return False
    if ":" in line:
        return False
    if TRANSITION_RE.match(line):
        return False
    if SCENE_HEADING_RE.match(line):
        return False
    if VISUAL_CUE_RE.match(line):
        return False
    clean = _normalize_character_name(line)
    return _is_valid_speaker(clean)


def _is_action_line(line: str) -> bool:
    if SCENE_HEADING_RE.match(line):
        return False
    if VISUAL_CUE_RE.match(line):
        return False
    if TRANSITION_RE.match(line):
        return False
    if _is_dialogue_line(line):
        return False
    return True


def _extract_location(heading: str) -> str:
    cleaned = heading.strip()
    # Remove leading scene number if present
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    # Remove INT./EXT. prefix
    cleaned = re.sub(r"^(INT/EXT\.|I/E\.|INT\.|EXT\.)\s*", "", cleaned, flags=re.IGNORECASE)
    # Take only the location part before " - DAY" / " - NIGHT" etc.
    parts = re.split(r"\s*-\s*", cleaned, maxsplit=1)
    return parts[0].strip() or cleaned


def _normalize_line(line: str) -> str:
    cleaned = line.strip()
    # Remove markdown bold/italic markers
    cleaned = re.sub(r"^[*_`]+", "", cleaned)
    cleaned = re.sub(r"[*_`]+$", "", cleaned)
    return cleaned.strip()