from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError


class DialogueLine(BaseModel):
    speaker: str = Field(min_length=1)
    line: str = Field(min_length=1)
    visual_cue: str = Field(min_length=1)
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    voice: Optional[str] = None


class SceneEntry(BaseModel):
    scene_id: int = Field(ge=1)
    location: str = Field(min_length=1)
    dialogue: List[DialogueLine] = Field(default_factory=list)
    characters: List[str] = Field(default_factory=list)
    duration: Optional[int] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    audio_file: Optional[str] = None
    bgm_file: Optional[str] = None
    mood: Optional[str] = None
    lines: Optional[List[DialogueLine]] = None


class SceneManifest(BaseModel):
    workflow_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    scenes: List[SceneEntry] = Field(min_length=1)
    total_scenes: int = Field(ge=1)
    total_duration_seconds: int = Field(ge=0)


class CharacterEntry(BaseModel):
    name: str = Field(min_length=1)
    personality: Optional[str] = ""
    appearance: Optional[str] = ""
    role: Optional[str] = "supporting"
    style_reference: Optional[str] = "Cinematic"
    first_appearance: Optional[int] = None
    dialogue_samples: Optional[List[Dict[str, Any]]] = None


class CharacterDB(BaseModel):
    workflow_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    characters: List[CharacterEntry] = Field(default_factory=list)
    total_characters: int = Field(ge=0)


class StoryCore(BaseModel):
    title: str = Field(min_length=1)
    logline: str = ""
    genre: str = "Drama"
    tone: str = "Dramatic"
    setting: str = ""
    time_period: str = ""
    themes: List[str] = Field(default_factory=list)
    acts: List[Dict[str, Any]] = Field(default_factory=list)
    protagonist: str = ""
    antagonist: Optional[str] = None
    world: str = ""


class StoryManifest(BaseModel):
    workflow_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    story: StoryCore


class TimingLine(BaseModel):
    speaker: str = Field(min_length=1)
    line: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    voice: str = ""
    visual_cue: str = ""


class TimingScene(BaseModel):
    scene_id: int = Field(ge=1)
    audio_file: str = Field(min_length=1)
    bgm_file: str = ""
    mood: str = "neutral"
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    lines: List[TimingLine] = Field(default_factory=list)


class TimingManifest(BaseModel):
    workflow_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    run_tag: str = Field(min_length=1)
    total_duration_ms: int = Field(ge=0)
    scenes: List[TimingScene] = Field(default_factory=list)


def validate_phase1_payloads(
    story_manifest: Dict[str, Any],
    scene_manifest: Dict[str, Any],
    character_db: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Strict schema enforcement for Phase 1 outputs.
    Raises ValidationError if any payload is malformed.
    """
    story_obj = StoryManifest.model_validate(story_manifest)
    scene_obj = SceneManifest.model_validate(scene_manifest)
    char_obj = CharacterDB.model_validate(character_db)
    return (
        story_obj.model_dump(),
        scene_obj.model_dump(),
        char_obj.model_dump(),
    )


def validate_timing_manifest_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strict validation for Phase 2 timing manifest shape.
    """
    obj = TimingManifest.model_validate(payload)
    return obj.model_dump()
