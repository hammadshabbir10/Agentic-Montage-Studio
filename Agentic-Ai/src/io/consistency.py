from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _core_name(text: str) -> str:
    return re.sub(r"[^A-Z0-9 ]+", "", text.upper()).strip()


def _name_match(target: str, candidates: List[str]) -> str | None:
    if not target:
        return None
    t = _core_name(target)
    if not t:
        return None
    for c in candidates:
        cc = _core_name(c)
        if t == cc or t in cc or cc in t:
            return c
    return None


def enforce_phase1_character_consistency(
    story_manifest: Dict[str, Any],
    scene_manifest: Dict[str, Any],
    character_db: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Reduce generation drift by reconciling:
    - speakers in scene dialogue
    - scene characters[]
    - character_db.characters[].name
    - story protagonist / antagonist references
    """
    warnings: List[str] = []

    scenes = scene_manifest.get("scenes", [])
    all_speakers: List[str] = []
    for scene in scenes:
        dialogue = scene.get("dialogue", [])
        speakers = []
        for d in dialogue:
            sp = _core_name(d.get("speaker", ""))
            if sp:
                d["speaker"] = sp
                if sp not in speakers:
                    speakers.append(sp)
                if sp not in all_speakers:
                    all_speakers.append(sp)
        scene["characters"] = speakers

    chars = character_db.get("characters", [])
    db_names: List[str] = []
    for c in chars:
        name = _core_name(c.get("name", ""))
        if name:
            c["name"] = name
            if name not in db_names:
                db_names.append(name)

    # Add any dialogue speakers missing from character_db
    missing = [s for s in all_speakers if s not in db_names]
    for name in missing:
        warnings.append(f"Added missing character '{name}' from dialogue into character_db.")
        chars.append(
            {
                "name": name,
                "personality": f"{name} appears in dialogue and should remain consistent.",
                "appearance": "Distinctive cinematic appearance.",
                "role": "supporting",
                "style_reference": "Cinematic",
                "dialogue_samples": [],
            }
        )
        db_names.append(name)

    character_db["characters"] = chars
    character_db["total_characters"] = len(chars)

    story = story_manifest.get("story", {})
    protagonist = story.get("protagonist", "")
    antagonist = story.get("antagonist", None)

    matched_protagonist = _name_match(protagonist, db_names)
    if not matched_protagonist and db_names:
        matched_protagonist = db_names[0]
        warnings.append(
            "Story protagonist drift detected; aligned protagonist to scene characters."
        )
    if matched_protagonist:
        story["protagonist"] = matched_protagonist

    if antagonist:
        matched_antagonist = _name_match(str(antagonist), db_names)
        if matched_antagonist:
            story["antagonist"] = matched_antagonist
        elif len(db_names) > 1:
            story["antagonist"] = db_names[1]
            warnings.append(
                "Story antagonist drift detected; aligned antagonist to scene characters."
            )
        else:
            story["antagonist"] = None

    story_manifest["story"] = story
    return story_manifest, scene_manifest, character_db, warnings
