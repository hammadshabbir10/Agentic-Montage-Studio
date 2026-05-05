"""
test_phase5.py  –  Phase 5 Unit Tests (Edit Agent & Undo)

Covers 15 test cases as required by the PDF:
  - 10+ edit query type classifications
  - State snapshot and revert
  - Version history with diffs
  - Image filter application
  - Asset restore on revert
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# ── Intent Classifier Tests ──────────────────────────────────────────────────

from src.agents.edit_intent_classifier import (
    EditIntent,
    classify_without_llm,
    _extract_scene_id,
    _extract_filter,
    _extract_tone,
)


class TestIntentClassification:
    """Test edit intent classification for 10+ query types."""

    def test_classify_change_voice_tone(self) -> None:
        """Audio target: change voice tone."""
        intent = classify_without_llm("Change voice tone to whispered")
        assert intent.target == "audio"
        assert intent.intent == "change_voice_tone"
        assert intent.confidence >= 0.9

    def test_classify_make_scene_darker(self) -> None:
        """Video_frame target: make scene darker."""
        intent = classify_without_llm("Make scene 1 darker")
        assert intent.target == "video_frame"
        assert intent.intent == "make_scene_darker"
        assert intent.parameters.get("scene_id") == 1

    def test_classify_add_background_music(self) -> None:
        """Audio target: add background music."""
        intent = classify_without_llm("Add background music with tense mood")
        assert intent.target == "audio"
        assert intent.intent == "add_background_music"
        assert intent.parameters.get("mood") == "tense"

    def test_classify_remove_subtitle(self) -> None:
        """Video target: remove subtitles."""
        intent = classify_without_llm("Remove the subtitles")
        assert intent.target == "video"
        assert intent.intent == "remove_subtitle"
        assert intent.parameters.get("subtitles") is False

    def test_classify_change_character_design(self) -> None:
        """Video_frame target: change character design."""
        intent = classify_without_llm("Change character design of Jack")
        assert intent.target == "video_frame"
        assert intent.intent == "change_character_design"

    def test_classify_speed_up_scene(self) -> None:
        """Video target: speed up scene."""
        intent = classify_without_llm("Speed up scene 2")
        assert intent.target == "video"
        assert intent.intent == "speed_up_scene"
        assert intent.parameters.get("scene_id") == 2

    def test_classify_regenerate_script(self) -> None:
        """Script target: regenerate the script."""
        intent = classify_without_llm("Regenerate the script")
        assert intent.target == "script"
        assert intent.intent == "regenerate_script"

    def test_classify_apply_sepia_filter(self) -> None:
        """Video_frame target: apply sepia filter."""
        intent = classify_without_llm("Apply sepia filter to scene 1")
        assert intent.target == "video_frame"
        assert intent.intent == "apply_filter"
        assert intent.parameters.get("filter") == "sepia"
        assert intent.parameters.get("scene_id") == 1

    def test_classify_adjust_volume(self) -> None:
        """Audio target: adjust volume."""
        intent = classify_without_llm("Increase the volume")
        assert intent.target == "audio"
        assert intent.intent == "adjust_volume"
        assert intent.parameters.get("adjustment") == "louder"

    def test_classify_change_scene_mood(self) -> None:
        """Audio target: change scene mood."""
        intent = classify_without_llm("Change the scene mood to happy")
        assert intent.target == "audio"
        assert intent.intent == "change_scene_mood"
        assert intent.parameters.get("mood") == "happy"

    def test_classify_make_scene_brighter(self) -> None:
        """Video_frame target: make scene brighter."""
        intent = classify_without_llm("Make scene 3 brighter")
        assert intent.target == "video_frame"
        assert intent.intent == "make_scene_brighter"
        assert intent.parameters.get("scene_id") == 3

    def test_classify_grayscale(self) -> None:
        """Video_frame target: black and white / grayscale."""
        intent = classify_without_llm("Make the scene black and white")
        assert intent.target == "video_frame"
        assert intent.intent == "apply_filter"
        assert intent.parameters.get("filter") == "grayscale"

    def test_classify_blur(self) -> None:
        """Video_frame target: blur."""
        intent = classify_without_llm("Blur the scene image")
        assert intent.target == "video_frame"
        assert intent.parameters.get("filter") == "blur"

    def test_classify_unknown_fallback(self) -> None:
        """Unknown query should fall back gracefully."""
        intent = classify_without_llm("xyzzy foobar baz")
        assert intent.intent == "unknown"
        assert intent.confidence <= 0.5


class TestExtractionHelpers:
    """Test extraction utility functions."""

    def test_extract_scene_id(self) -> None:
        assert _extract_scene_id("Make scene 3 darker") == 3
        assert _extract_scene_id("No scene mentioned") is None

    def test_extract_filter(self) -> None:
        assert _extract_filter("apply sepia filter") == "sepia"
        assert _extract_filter("use blur effect") == "blur"

    def test_extract_tone(self) -> None:
        assert _extract_tone("change to whispered voice") == "whispered"
        assert _extract_tone("no tone here") == "default"


# ── State Versioning Tests ───────────────────────────────────────────────────

from src.state_versioning import StateManager


class TestStateVersioning:
    """Test state snapshot, revert, and history."""

    @pytest.fixture
    def tmp_state_dir(self, tmp_path: Path) -> Path:
        """Create a temporary directory for state versioning tests."""
        return tmp_path / "state_versions"

    @pytest.fixture
    def sm(self, tmp_state_dir: Path) -> StateManager:
        """Create a StateManager with a temp directory."""
        return StateManager(base_dir=tmp_state_dir)

    def test_state_snapshot_and_revert(self, sm: StateManager) -> None:
        """Test basic snapshot creation and revert."""
        # Create initial state
        state_v1 = {
            "story_manifest": {"title": "Original Story"},
            "scene_manifest": {"scenes": [{"scene_id": 1}]},
        }
        v1 = sm.snapshot(
            state_json=state_v1,
            description="Initial output",
            target="pipeline",
        )
        assert v1 == 1
        assert sm.current_version() == 1

        # Create a modified state
        state_v2 = {
            "story_manifest": {"title": "Modified Story"},
            "scene_manifest": {"scenes": [{"scene_id": 1}, {"scene_id": 2}]},
        }
        v2 = sm.snapshot(
            state_json=state_v2,
            description="Added scene 2",
            target="script",
        )
        assert v2 == 2

        # Revert to v1
        restored = sm.revert(1)
        assert restored["story_manifest"]["title"] == "Original Story"
        assert len(restored["scene_manifest"]["scenes"]) == 1

        # A revert should create a new version (v3)
        assert sm.current_version() == 3

    def test_state_history(self, sm: StateManager) -> None:
        """Test version history with diff summaries."""
        sm.snapshot(
            state_json={"key": "value1"},
            description="Version 1",
            target="pipeline",
        )
        sm.snapshot(
            state_json={"key": "value2"},
            description="Version 2",
            target="audio",
        )

        history = sm.history()
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[0]["diff_summary"] == "Initial pipeline output"
        assert history[1]["version"] == 2
        assert "State changed" in history[1]["diff_summary"]

    def test_revert_restores_assets(self, sm: StateManager, tmp_path: Path) -> None:
        """Test that revert restores asset files."""
        # Create a temp asset file
        asset_file = tmp_path / "test_asset.txt"
        asset_file.write_text("original content", encoding="utf-8")

        # Snapshot with asset
        sm.snapshot(
            state_json={"phase": "initial"},
            asset_paths=[str(asset_file)],
            description="With asset",
            target="pipeline",
        )

        # Modify the asset
        asset_file.write_text("modified content", encoding="utf-8")

        # Snapshot modified state
        sm.snapshot(
            state_json={"phase": "modified"},
            asset_paths=[str(asset_file)],
            description="Modified asset",
            target="video_frame",
        )

        # Revert to v1
        sm.revert(1)

        # Asset should be restored
        assert asset_file.read_text(encoding="utf-8") == "original content"

    def test_get_version_state(self, sm: StateManager) -> None:
        """Test retrieving state for a specific version."""
        sm.snapshot(
            state_json={"data": "test123"},
            description="Test version",
            target="pipeline",
        )
        state = sm.get_version_state(1)
        assert state["data"] == "test123"

    def test_get_version_not_found(self, sm: StateManager) -> None:
        """Test error when version doesn't exist."""
        with pytest.raises(FileNotFoundError):
            sm.get_version_state(999)


# ── Image Filter Tests ───────────────────────────────────────────────────────

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

if HAS_CV2:
    from src.utils.image_filters import (
        apply_brightness,
        apply_sepia,
        apply_grayscale,
        apply_blur,
        apply_sharpen,
        apply_filter_chain,
        get_available_filters,
    )


@pytest.mark.skipif(not HAS_CV2, reason="opencv-python-headless not installed")
class TestImageFilters:
    """Test OpenCV image filter application."""

    @pytest.fixture
    def test_image(self, tmp_path: Path) -> str:
        """Create a simple test image."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)  # grey image
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)
        return path

    def test_image_filter_brightness(self, test_image: str) -> None:
        """Test brightness adjustment."""
        output = tmp_path_for_output(test_image, "bright")
        result = apply_brightness(test_image, factor=1.5, output_path=output)
        assert Path(result).exists()
        # Brighter image should have higher mean pixel value
        orig = cv2.imread(test_image)
        mod = cv2.imread(result)
        assert mod.mean() > orig.mean()

    def test_image_filter_sepia(self, test_image: str) -> None:
        """Test sepia filter application."""
        output = tmp_path_for_output(test_image, "sepia")
        result = apply_sepia(test_image, output_path=output)
        assert Path(result).exists()
        # Sepia should change the image
        img = cv2.imread(result)
        assert img is not None

    def test_image_filter_grayscale(self, test_image: str) -> None:
        """Test grayscale conversion."""
        output = tmp_path_for_output(test_image, "gray")
        result = apply_grayscale(test_image, output_path=output)
        assert Path(result).exists()

    def test_image_filter_blur(self, test_image: str) -> None:
        """Test blur filter."""
        output = tmp_path_for_output(test_image, "blur")
        result = apply_blur(test_image, kernel_size=15, output_path=output)
        assert Path(result).exists()

    def test_image_filter_chain(self, test_image: str) -> None:
        """Test applying multiple filters in sequence."""
        output = tmp_path_for_output(test_image, "chain")
        result = apply_filter_chain(
            test_image,
            filters=[
                {"name": "sepia"},
                {"name": "brightness", "factor": 0.8},
            ],
            output_path=output,
        )
        assert Path(result).exists()

    def test_available_filters(self) -> None:
        """Test that filter registry returns expected filters."""
        filters = get_available_filters()
        assert "sepia" in filters
        assert "brightness" in filters
        assert "blur" in filters
        assert "grayscale" in filters
        assert len(filters) >= 7


# ── EditIntent Pydantic Model Tests ──────────────────────────────────────────

class TestEditIntentModel:
    """Test Pydantic EditIntent validation."""

    def test_valid_intent(self) -> None:
        intent = EditIntent(
            intent="apply_filter",
            target="video_frame",
            scope="scene:1",
            parameters={"filter": "sepia"},
            confidence=0.95,
        )
        assert intent.intent == "apply_filter"
        assert intent.confidence == 0.95

    def test_intent_defaults(self) -> None:
        intent = EditIntent(intent="test", target="audio")
        assert intent.scope == "all"
        assert intent.parameters == {}
        assert intent.confidence == 1.0

    def test_intent_serialisation(self) -> None:
        intent = EditIntent(
            intent="change_voice_tone",
            target="audio",
            parameters={"tone": "whispered"},
        )
        data = intent.model_dump()
        assert data["intent"] == "change_voice_tone"
        assert data["parameters"]["tone"] == "whispered"

        # Round-trip
        restored = EditIntent(**data)
        assert restored == intent


# ── Helper ───────────────────────────────────────────────────────────────────

def tmp_path_for_output(original: str, suffix: str) -> str:
    """Generate a temp output path next to the original."""
    p = Path(original)
    return str(p.parent / f"{p.stem}_{suffix}{p.suffix}")
