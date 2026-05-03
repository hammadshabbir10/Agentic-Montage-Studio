"""
Phase 3 unit tests.

Coverage
--------
1. Input contract validation (positive and negative cases).
2. Scene plan cross-join from scene_manifest + timing_manifest.
3. Image prompt construction (location, mood, characters, style anchor).
4. Image backend fallback (HF fails -> Pollinations succeeds).
5. SRT generation correctness from timing_manifest.

Run with:
    python -m unittest tests.test_phase3 -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.io.phase3_contracts import (
    Phase3ValidationError,
    build_scene_plans,
    validate_phase3_inputs,
    validate_scene_manifest,
    validate_timing_manifest,
)
from src.agents import scene_visualizer
from src.utils import video_compose


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_scene_manifest():
    return {
        "scenes": [
            {
                "scene_id": 1,
                "location": "TIME LAB",
                "dialogue": [
                    {"speaker": "DR. VICTORIA",
                     "line": "We are ready.",
                     "visual_cue": "Medium shot of DR. VICTORIA, intense."},
                ],
                "characters": ["DR. VICTORIA"],
                "duration": 6,
            }
        ]
    }


def _write_dummy(path: Path, content: bytes = b"x") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _make_timing_manifest(audio_path: str, bgm_path: str = ""):
    return {
        "scenes": [
            {
                "scene_id": 1,
                "audio_file": audio_path,
                "bgm_file":   bgm_path,
                "mood":       "tense",
                "start_ms":   0,
                "end_ms":     6000,
                "duration_ms": 6000,
                "lines": [
                    {"speaker": "DR. VICTORIA",
                     "voice":   "en-US-JennyNeural",
                     "line":    "We are ready.",
                     "start_ms": 0,
                     "end_ms":   2000,
                     "duration_ms": 2000},
                ],
            }
        ]
    }


def _make_character_db():
    return {
        "characters": [
            {
                "name": "DR. VICTORIA",
                "appearance": "Mid-30s scientist in tailored lab coat",
                "personality": "Confident",
                "role": "supporting",
                "style_reference": "retro-futuristic",
                "first_appearance": 1,
                "dialogue_samples": [],
            }
        ],
        "total_characters": 1,
    }


# ── Validation tests ────────────────────────────────────────────────────────

class TestValidation(unittest.TestCase):

    def test_scene_manifest_missing_scenes(self):
        with self.assertRaises(Phase3ValidationError):
            validate_scene_manifest({})

    def test_scene_manifest_empty_scenes(self):
        with self.assertRaises(Phase3ValidationError):
            validate_scene_manifest({"scenes": []})

    def test_scene_manifest_missing_required_keys(self):
        bad = {"scenes": [{"scene_id": 1}]}
        with self.assertRaises(Phase3ValidationError):
            validate_scene_manifest(bad)

    def test_timing_manifest_missing_audio_file(self):
        with TemporaryDirectory() as tmp:
            timing = _make_timing_manifest(audio_path=str(Path(tmp) / "missing.mp3"))
            with self.assertRaises(Phase3ValidationError):
                validate_timing_manifest(timing)

    def test_timing_manifest_valid(self):
        with TemporaryDirectory() as tmp:
            audio = _write_dummy(Path(tmp) / "a.mp3")
            timing = _make_timing_manifest(audio_path=audio)
            # should not raise
            validate_timing_manifest(timing)

    def test_validate_phase3_inputs_full(self):
        with TemporaryDirectory() as tmp:
            audio = _write_dummy(Path(tmp) / "a.mp3")
            plans = validate_phase3_inputs(
                scene_manifest=_make_scene_manifest(),
                timing_manifest=_make_timing_manifest(audio_path=audio),
                character_db=_make_character_db(),
            )
            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0].location, "TIME LAB")
            self.assertEqual(plans[0].mood, "tense")
            self.assertGreater(plans[0].duration_sec, 0)


class TestPlanBuilder(unittest.TestCase):

    def test_cross_join_picks_timing_fields(self):
        with TemporaryDirectory() as tmp:
            audio = _write_dummy(Path(tmp) / "a.mp3")
            plans = build_scene_plans(
                _make_scene_manifest(),
                _make_timing_manifest(audio_path=audio),
            )
            p = plans[0]
            self.assertEqual(p.scene_id, 1)
            self.assertEqual(p.audio_file, audio)
            self.assertEqual(p.speakers, ["DR. VICTORIA"])
            self.assertEqual(p.visual_cues[0].split()[0], "Medium")

    def test_missing_timing_for_scene_raises(self):
        scene = _make_scene_manifest()
        timing = {"scenes": []}
        with self.assertRaises(Phase3ValidationError):
            build_scene_plans(scene, timing)


# ── Prompt builder tests ────────────────────────────────────────────────────

class TestPromptBuilder(unittest.TestCase):

    def test_prompt_contains_location_mood_character(self):
        with TemporaryDirectory() as tmp:
            audio = _write_dummy(Path(tmp) / "a.mp3")
            plan = build_scene_plans(
                _make_scene_manifest(),
                _make_timing_manifest(audio_path=audio),
            )[0]
            prompt = scene_visualizer.build_scene_prompt(plan, _make_character_db())
            self.assertIn("Time Lab", prompt)
            self.assertIn("tense", prompt)
            self.assertIn("Dr. Victoria", prompt)
            self.assertIn("cinematic", prompt.lower())


# ── Backend fallback tests ──────────────────────────────────────────────────

class TestBackendFallback(unittest.TestCase):

    def test_hf_fails_then_pollinations_succeeds(self):
        with TemporaryDirectory() as tmp:
            audio = _write_dummy(Path(tmp) / "a.mp3")
            plan = build_scene_plans(
                _make_scene_manifest(),
                _make_timing_manifest(audio_path=audio),
            )[0]

            def _hf_fail(prompt, width, height, out_path, seed=None, retries=2):
                return False, "hf:failure"

            def _poll_ok(prompt, width, height, out_path, seed=None):
                out_path.write_bytes(b"\x89PNG")
                return True, "pollinations"

            with patch.object(scene_visualizer, "_generate_via_hf", side_effect=_hf_fail) as m_hf, \
                 patch.object(scene_visualizer, "_generate_via_pollinations", side_effect=_poll_ok) as m_poll:
                images_dir = Path(tmp) / "images"
                result = scene_visualizer.generate_scene_image(
                    plan, _make_character_db(),
                    images_dir=images_dir,
                    backend="auto",
                    quality="fast",
                    seed=42,
                    use_cache=False,
                )
                self.assertEqual(result["backend"], "pollinations")
                self.assertTrue(Path(result["image_path"]).exists())
                m_hf.assert_called_once()
                m_poll.assert_called_once()

    def test_explicit_pollinations_backend_skips_hf(self):
        with TemporaryDirectory() as tmp:
            audio = _write_dummy(Path(tmp) / "a.mp3")
            plan = build_scene_plans(
                _make_scene_manifest(),
                _make_timing_manifest(audio_path=audio),
            )[0]

            def _poll_ok(prompt, width, height, out_path, seed=None):
                out_path.write_bytes(b"\x89PNG")
                return True, "pollinations"

            with patch.object(scene_visualizer, "_generate_via_hf") as m_hf, \
                 patch.object(scene_visualizer, "_generate_via_pollinations", side_effect=_poll_ok):
                scene_visualizer.generate_scene_image(
                    plan, _make_character_db(),
                    images_dir=Path(tmp) / "images",
                    backend="pollinations",
                    quality="fast",
                    use_cache=False,
                )
                m_hf.assert_not_called()

    def test_image_cache_hit_skips_backends(self):
        with TemporaryDirectory() as tmp:
            audio = _write_dummy(Path(tmp) / "a.mp3")
            plan = build_scene_plans(
                _make_scene_manifest(),
                _make_timing_manifest(audio_path=audio),
            )[0]

            images_dir = Path(tmp) / "images"

            # First call: populate cache
            def _poll_ok(prompt, width, height, out_path, seed=None):
                out_path.write_bytes(b"\x89PNG")
                return True, "pollinations"

            with patch.object(scene_visualizer, "_generate_via_hf"), \
                 patch.object(scene_visualizer, "_generate_via_pollinations", side_effect=_poll_ok):
                first = scene_visualizer.generate_scene_image(
                    plan, _make_character_db(),
                    images_dir=images_dir,
                    backend="pollinations",
                    quality="fast",
                    seed=42,
                )

            # Second call with the same prompt/seed: backend MUST NOT be called
            with patch.object(scene_visualizer, "_generate_via_hf") as m_hf, \
                 patch.object(scene_visualizer, "_generate_via_pollinations") as m_poll:
                second = scene_visualizer.generate_scene_image(
                    plan, _make_character_db(),
                    images_dir=images_dir,
                    backend="pollinations",
                    quality="fast",
                    seed=42,
                )
                m_hf.assert_not_called()
                m_poll.assert_not_called()
                self.assertEqual(second["backend"], "cache")
                self.assertEqual(first["image_path"], second["image_path"])


# ── SRT and ffmpeg helper tests ─────────────────────────────────────────────

class TestSubtitlesAndProfiles(unittest.TestCase):

    def test_ms_to_srt_format(self):
        self.assertEqual(video_compose._ms_to_srt(0), "00:00:00,000")
        self.assertEqual(video_compose._ms_to_srt(1234), "00:00:01,234")
        self.assertEqual(video_compose._ms_to_srt(3_661_500), "01:01:01,500")

    def test_build_srt_creates_file_with_lines(self):
        with TemporaryDirectory() as tmp:
            audio = _write_dummy(Path(tmp) / "a.mp3")
            timing = _make_timing_manifest(audio_path=audio)
            srt_out = Path(tmp) / "subs.srt"
            video_compose.build_srt(timing, srt_out)
            content = srt_out.read_text(encoding="utf-8")
            self.assertIn("DR. VICTORIA", content)
            self.assertIn("--> ", content)
            self.assertIn("00:00:00,000", content)

    def test_get_profile_returns_known_or_balanced(self):
        prof = video_compose.get_profile("balanced")
        self.assertEqual((prof.width, prof.height), (1280, 720))
        prof_unknown = video_compose.get_profile("does-not-exist")
        self.assertEqual((prof_unknown.width, prof_unknown.height), (1280, 720))


if __name__ == "__main__":
    unittest.main()
