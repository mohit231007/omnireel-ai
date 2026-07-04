"""Global orchestration for OmniReel AI.

This module connects the Python asset pipeline to the compiled Rust compositor. It
keeps the two boundaries intentionally clean: Python produces files and a manifest;
Rust consumes file paths and performs CPU-heavy video/subtitle processing.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .models_pipeline import AssetManifest, OmniReelPipeline, PipelineConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for Python-to-Rust execution."""

    pipeline: PipelineConfig
    project_root: Path
    rust_binary: Optional[Path] = None
    build_release: bool = True
    skip_python_generation: bool = False
    existing_manifest: Optional[Path] = None
    rust_threads: Optional[int] = None
    rust_crf: int = 18
    audio_bitrate: str = "192k"


class OmniReelOrchestrator:
    """Runs asset generation and invokes the Rust compositor via subprocess."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.project_root = config.project_root.resolve()
        if not self.project_root.exists():
            raise FileNotFoundError(f"Project root does not exist: {self.project_root}")

    def run(self, topic: str) -> AssetManifest:
        """Execute the full local pipeline and return the final manifest."""
        self._log_environment()
        manifest = self._resolve_manifest(topic)
        engine = self._resolve_or_build_engine()
        self._run_rust_compositor(engine, manifest)
        return manifest

    def _resolve_manifest(self, topic: str) -> AssetManifest:
        if self.config.skip_python_generation:
            if not self.config.existing_manifest:
                raise ValueError("skip_python_generation=True requires existing_manifest.")
            LOGGER.info("Loading existing manifest from %s", self.config.existing_manifest)
            return _manifest_from_json(self.config.existing_manifest)

        pipeline = OmniReelPipeline(self.config.pipeline)
        return pipeline.run(topic)

    def _resolve_or_build_engine(self) -> Path:
        if self.config.rust_binary:
            binary = self.config.rust_binary.resolve()
            if not binary.exists():
                raise FileNotFoundError(f"Configured Rust binary not found: {binary}")
            return binary

        if self.config.build_release:
            self._cargo_build_release()
            binary = self.project_root / "target" / "release" / _engine_binary_name()
        else:
            binary = self.project_root / "target" / "debug" / _engine_binary_name()

        if not binary.exists():
            raise FileNotFoundError(
                f"Rust engine binary not found after build: {binary}. "
                "Check cargo output for build errors."
            )
        return binary

    def _cargo_build_release(self) -> None:
        cargo = shutil.which("cargo")
        if not cargo:
            raise FileNotFoundError("Rust cargo executable not found on PATH.")

        command = [cargo, "build", "--release"]
        LOGGER.info("Building Rust compositor: %s", " ".join(command))
        _run_command(command, cwd=self.project_root, label="cargo build --release")

    def _run_rust_compositor(self, engine: Path, manifest: AssetManifest) -> None:
        for path, label in [
            (manifest.raw_video, "raw video"),
            (manifest.audio, "audio"),
            (manifest.subtitles_json, "subtitle JSON"),
        ]:
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing {label} required by Rust compositor: {path}")

        command: List[str] = [
            str(engine),
            "--input-video",
            str(manifest.raw_video),
            "--audio",
            str(manifest.audio),
            "--subtitles",
            str(manifest.subtitles_json),
            "--output",
            str(manifest.final_video),
            "--fps",
            str(manifest.fps),
            "--crf",
            str(self.config.rust_crf),
            "--audio-bitrate",
            self.config.audio_bitrate,
        ]
        if manifest.font_path:
            command.extend(["--font", str(manifest.font_path)])
        if self.config.rust_threads:
            command.extend(["--threads", str(self.config.rust_threads)])

        LOGGER.info("Invoking Rust compositor: %s", " ".join(command))
        _run_command(command, cwd=self.project_root, label="omnireel-ai-engine")

        if not manifest.final_video.exists() or manifest.final_video.stat().st_size == 0:
            raise FileNotFoundError(f"Rust compositor did not create final video: {manifest.final_video}")
        LOGGER.info("Final OmniReel video created: %s", manifest.final_video)

    def _log_environment(self) -> None:
        LOGGER.info("Python executable: %s", sys.executable)
        LOGGER.info("Python version: %s", sys.version.replace("\n", " "))
        LOGGER.info("OS: %s %s", platform.system(), platform.release())
        LOGGER.info("Project root: %s", self.project_root)
        LOGGER.info("Output dir: %s", self.config.pipeline.output_dir)


def _engine_binary_name() -> str:
    return "omnireel-ai-engine.exe" if os.name == "nt" else "omnireel-ai-engine"


def _run_command(command: Sequence[str], cwd: Path, label: str) -> None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} executable not found: {command[0]}") from exc

    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout[-6000:]}\n"
            f"STDERR:\n{completed.stderr[-6000:]}"
        )

    if completed.stdout.strip():
        LOGGER.debug("%s stdout: %s", label, completed.stdout[-3000:])
    if completed.stderr.strip():
        LOGGER.debug("%s stderr: %s", label, completed.stderr[-3000:])


def _manifest_from_json(path: Path) -> AssetManifest:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ["output_dir", "plan_json", "base_image", "audio", "raw_video", "subtitles_json", "final_video", "fps"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Manifest missing keys: {missing}")

    return AssetManifest(
        output_dir=Path(payload["output_dir"]),
        plan_json=Path(payload["plan_json"]),
        base_image=Path(payload["base_image"]),
        audio=Path(payload["audio"]),
        raw_video=Path(payload["raw_video"]),
        subtitles_json=Path(payload["subtitles_json"]),
        final_video=Path(payload["final_video"]),
        font_path=Path(payload["font_path"]) if payload.get("font_path") else None,
        fps=float(payload["fps"]),
    )
