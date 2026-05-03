from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.io.consistency import enforce_phase1_character_consistency
from src.io.pydantic_schemas import validate_phase1_payloads


def _fixtures():
    story_manifest = {
        "workflow_id": "workflow_1",
        "timestamp": "2026-05-04T00:00:00",
        "story": {
            "title": "Time Drift",
            "logline": "A scientist races against time.",
            "genre": "Sci-Fi",
            "tone": "tense",
            "setting": "lab",
            "time_period": "future",
            "themes": ["time", "risk"],
            "acts": [],
            "protagonist": "Dr. Ethan Slade",  # drifted name (not in scene speakers)
            "antagonist": "Unknown Agency",
            "world": "experimental world",
        },
    }
    scene_manifest = {
        "workflow_id": "workflow_2",
        "timestamp": "2026-05-04T00:00:00",
        "scenes": [
            {
                "scene_id": 1,
                "location": "TIME LAB",
                "dialogue": [
                    {"speaker": "DR. VICTORIA", "line": "Start the jump.", "visual_cue": "Close-up."},
                    {"speaker": "DR. REED", "line": "Stabilizers green.", "visual_cue": "Wide shot."},
                ],
                "characters": ["DR. VICTORIA", "DR. REED"],
                "duration": 8,
            }
        ],
        "total_scenes": 1,
        "total_duration_seconds": 8,
    }
    character_db = {
        "workflow_id": "workflow_3",
        "timestamp": "2026-05-04T00:00:00",
        "characters": [
            {
                "name": "DR. VICTORIA",
                "personality": "Bold",
                "appearance": "Scientist",
                "role": "supporting",
                "style_reference": "Cinematic",
                "dialogue_samples": [],
            }
        ],
        "total_characters": 1,
    }
    return story_manifest, scene_manifest, character_db


class TestPhase1ConsistencyAndSchema(unittest.TestCase):
    def test_consistency_adds_missing_characters(self):
        story, scene, char_db = _fixtures()
        _, _, fixed_char_db, warnings = enforce_phase1_character_consistency(
            story, scene, char_db
        )
        names = [c["name"] for c in fixed_char_db["characters"]]
        self.assertIn("DR REED", names)
        self.assertTrue(any("Added missing character" in w for w in warnings))

    def test_consistency_aligns_protagonist(self):
        story, scene, char_db = _fixtures()
        fixed_story, _, fixed_char_db, _ = enforce_phase1_character_consistency(
            story, scene, char_db
        )
        names = [c["name"] for c in fixed_char_db["characters"]]
        self.assertIn(fixed_story["story"]["protagonist"], names)

    def test_phase1_schema_validation_passes_after_consistency(self):
        story, scene, char_db = _fixtures()
        fixed_story, fixed_scene, fixed_char_db, _ = enforce_phase1_character_consistency(
            story, scene, char_db
        )
        out_story, out_scene, out_char = validate_phase1_payloads(
            fixed_story, fixed_scene, fixed_char_db
        )
        self.assertEqual(out_scene["total_scenes"], 1)
        self.assertGreaterEqual(out_char["total_characters"], 2)

    def test_phase1_schema_validation_fails_on_bad_payload(self):
        with self.assertRaises(ValidationError):
            validate_phase1_payloads(
                {"workflow_id": "", "timestamp": "", "story": {}},
                {"workflow_id": "x", "timestamp": "y", "scenes": [], "total_scenes": 0, "total_duration_seconds": 0},
                {"workflow_id": "x", "timestamp": "y", "characters": [], "total_characters": 0},
            )


if __name__ == "__main__":
    unittest.main()
