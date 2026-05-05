"""
state_versioning.py  –  Phase 5 State Versioning & Undo System

Implements the PDF-specified StateManager API:
    StateManager.snapshot(version, state_json, asset_paths)  →  persisted
    StateManager.revert(version)   →  restores assets + state
    StateManager.history()         →  returns list of all versions with diff summary

Storage layout
--------------
data/state_versions/
    version_log.json          ← append-only list of version records
    v1/
        state.json            ← full pipeline state at this version
        assets/               ← copies of generated asset files
    v2/
        ...
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_BASE = Path("data/state_versions")


class StateManager:
    """Append-only state versioning with full snapshot / revert / history."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else _DEFAULT_BASE
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.base_dir / "version_log.json"

    # ── Public API ────────────────────────────────────────────────────────

    def snapshot(
        self,
        state_json: Dict[str, Any],
        asset_paths: List[str] | None = None,
        description: str = "",
        target: str = "pipeline",
    ) -> int:
        """
        Persist a state snapshot.  Returns the new version number.

        Parameters
        ----------
        state_json : dict
            The full pipeline JSON state at this moment.
        asset_paths : list[str], optional
            Absolute or relative paths to asset files (audio, images, video)
            that should be copied into the snapshot.
        description : str
            Human-readable summary of what happened (e.g. "initial generation",
            "applied sepia filter to scene 1").
        target : str
            Which component produced this version (pipeline | audio | video_frame | video | script).
        """
        version = self.current_version() + 1
        version_dir = self.base_dir / f"v{version}"
        assets_dir = version_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # ── Save state JSON ──────────────────────────────────────────────
        state_path = version_dir / "state.json"
        # Strip non-serialisable objects (tool_client, memory_store, etc.)
        clean_state = _strip_non_serialisable(state_json)
        state_path.write_text(
            json.dumps(clean_state, indent=2, default=str), encoding="utf-8"
        )

        # ── Copy asset files ─────────────────────────────────────────────
        asset_manifest: List[Dict[str, str]] = []
        for src_str in asset_paths or []:
            src = Path(src_str)
            if not src.exists():
                continue
            dest = assets_dir / src.name
            # Avoid self-copy if asset is already inside the version dir
            if dest.resolve() != src.resolve():
                shutil.copy2(str(src), str(dest))
            asset_manifest.append({
                "original_path": str(src),
                "snapshot_path": str(dest),
                "filename": src.name,
            })

        # ── Compute state hash ───────────────────────────────────────────
        state_hash = hashlib.sha256(
            json.dumps(clean_state, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        # ── Append to log ────────────────────────────────────────────────
        record = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description or f"Version {version}",
            "target": target,
            "state_hash": state_hash,
            "asset_count": len(asset_manifest),
            "asset_manifest": asset_manifest,
        }
        log = self._read_log()
        log.append(record)
        self._write_log(log)

        return version

    def revert(self, version: int) -> Dict[str, Any]:
        """
        Restore the pipeline state and assets from the given version.

        Returns the restored state dict.
        """
        version_dir = self.base_dir / f"v{version}"
        state_path = version_dir / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Version {version} not found at {version_dir}")

        restored_state = json.loads(state_path.read_text(encoding="utf-8"))

        # ── Restore assets to their original locations ───────────────────
        log = self._read_log()
        record = next((r for r in log if r["version"] == version), None)
        if record:
            for asset in record.get("asset_manifest", []):
                snapshot_path = Path(asset["snapshot_path"])
                original_path = Path(asset["original_path"])
                if snapshot_path.exists():
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(snapshot_path), str(original_path))

        # ── Also restore the state JSON files back to data/ ──────────────
        self._restore_state_files(restored_state)

        # ── Log the revert as a new version ──────────────────────────────
        revert_version = self.current_version() + 1
        revert_dir = self.base_dir / f"v{revert_version}"
        revert_dir.mkdir(parents=True, exist_ok=True)
        revert_state_path = revert_dir / "state.json"
        shutil.copy2(str(state_path), str(revert_state_path))

        # Copy assets from the reverted version
        src_assets = version_dir / "assets"
        dest_assets = revert_dir / "assets"
        if src_assets.exists():
            shutil.copytree(str(src_assets), str(dest_assets), dirs_exist_ok=True)

        revert_record = {
            "version": revert_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": f"Reverted to version {version}",
            "target": "revert",
            "state_hash": record["state_hash"] if record else "",
            "asset_count": record["asset_count"] if record else 0,
            "asset_manifest": record.get("asset_manifest", []) if record else [],
            "reverted_from": version,
        }
        log.append(revert_record)
        self._write_log(log)

        return restored_state

    def history(self) -> List[Dict[str, Any]]:
        """
        Return the complete version history with diff summaries.

        Each entry includes a `diff_summary` field describing what changed
        from the previous version.
        """
        log = self._read_log()
        result: List[Dict[str, Any]] = []

        for i, record in enumerate(log):
            entry = {
                "version": record["version"],
                "timestamp": record["timestamp"],
                "description": record["description"],
                "target": record["target"],
                "state_hash": record["state_hash"],
                "asset_count": record.get("asset_count", 0),
            }

            if "reverted_from" in record:
                entry["diff_summary"] = f"Reverted to version {record['reverted_from']}"
                entry["reverted_from"] = record["reverted_from"]
            elif i == 0:
                entry["diff_summary"] = "Initial pipeline output"
            else:
                prev = log[i - 1]
                if record["state_hash"] == prev["state_hash"]:
                    entry["diff_summary"] = "No state change (assets only)"
                else:
                    entry["diff_summary"] = (
                        f"State changed (target: {record['target']}). "
                        f"{record.get('description', '')}"
                    )

            result.append(entry)

        return result

    def current_version(self) -> int:
        """Return the latest version number, or 0 if no snapshots exist."""
        log = self._read_log()
        if not log:
            return 0
        return max(r["version"] for r in log)

    def get_version_state(self, version: int) -> Dict[str, Any]:
        """Load and return the state JSON for a specific version."""
        state_path = self.base_dir / f"v{version}" / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Version {version} state not found")
        return json.loads(state_path.read_text(encoding="utf-8"))

    def get_version_assets(self, version: int) -> List[Dict[str, str]]:
        """Return the asset manifest for a specific version."""
        log = self._read_log()
        record = next((r for r in log if r["version"] == version), None)
        if not record:
            raise FileNotFoundError(f"Version {version} not in log")
        return record.get("asset_manifest", [])

    # ── Private helpers ───────────────────────────────────────────────────

    def _read_log(self) -> List[Dict[str, Any]]:
        if not self._log_path.exists():
            return []
        try:
            return json.loads(self._log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _write_log(self, log: List[Dict[str, Any]]) -> None:
        self._log_path.write_text(
            json.dumps(log, indent=2, default=str), encoding="utf-8"
        )

    def _restore_state_files(self, state: Dict[str, Any]) -> None:
        """Write state components back to data/ working directory."""
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)

        mappings = {
            "story_manifest": "story_manifest_auto.json",
            "scene_manifest": "scene_manifest_auto.json",
            "character_db": "character_db_auto.json",
        }
        for key, filename in mappings.items():
            if key in state and state[key]:
                (data_dir / filename).write_text(
                    json.dumps(state[key], indent=2, default=str), encoding="utf-8"
                )


def _strip_non_serialisable(obj: Any) -> Any:
    """
    Recursively strip non-JSON-serialisable values (ToolClient, MemoryStore, etc.)
    so the state can be persisted to disk.
    """
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if k in ("tool_client", "memory_store"):
                continue
            try:
                json.dumps(v, default=str)
                clean[k] = _strip_non_serialisable(v)
            except (TypeError, ValueError, OverflowError):
                continue
        return clean
    if isinstance(obj, (list, tuple)):
        return [_strip_non_serialisable(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
