"""Local AI asset generation for OmniReel AI.

Python owns model orchestration only:
- Local LLM planning through Ollama.
- Local Diffusers image generation from an already-downloaded model path.
- Offline TTS through pyttsx3 or a user-provided Piper command.
- Local animation through either a user-provided command or a deterministic
  procedural motion backend for local product demos.
- Optional local Whisper/whisper.cpp timestamp extraction.

No remote APIs are called by this module. Any backend that requires a network
service is either rejected or treated as a locality violation.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

LOGGER = logging.getLogger(__name__)

TTSBackend = Literal["pyttsx3", "piper"]
DiffusionOptimization = Literal["auto", "fp16", "bf16", "fp32", "int8"]


class OmniReelError(RuntimeError):
    """Base exception for recoverable OmniReel pipeline failures."""


class LocalityViolation(OmniReelError):
    """Raised when a configured backend would cross the local machine boundary."""


class ExternalCommandError(OmniReelError):
    """Raised when a local subprocess exits unsuccessfully."""


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for local model orchestration."""

    output_dir: Path
    ollama_model: str
    diffusion_model_path: Path
    liveportrait_cmd: str
    whisper_cmd: Optional[str] = None
    font_path: Optional[Path] = None
    tts_backend: TTSBackend = "pyttsx3"
    piper_cmd: Optional[str] = None
    optimization: DiffusionOptimization = "auto"
    width: int = 768
    height: int = 768
    inference_steps: int = 28
    guidance_scale: float = 7.0
    seed: Optional[int] = 42
    fps: float = 30.0
    duration_seconds: float = 3.0
    allow_remote_backends: bool = False
    qa_static_assets: bool = False


@dataclass(frozen=True)
class ScenePlan:
    """Validated LLM output used by downstream generation steps."""

    title: str
    visual_prompt: str
    negative_prompt: str
    dialogue: str
    style_notes: str


@dataclass(frozen=True)
class AssetManifest:
    """File contract between Python and the Rust compositor."""

    output_dir: Path
    plan_json: Path
    base_image: Path
    audio: Path
    raw_video: Path
    subtitles_json: Path
    final_video: Path
    font_path: Optional[Path]
    fps: float

    def to_json_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


class OmniReelPipeline:
    """End-to-end local asset generator used before Rust compositing."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_locality()

    def _validate_locality(self) -> None:
        """Fail fast if a configured backend would require external network I/O."""
        if self.config.allow_remote_backends:
            raise LocalityViolation("allow_remote_backends=True is not permitted in OmniReel AI.")

        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
        if not _is_loopback_url(ollama_host):
            raise LocalityViolation(f"OLLAMA_HOST must be loopback/local, got: {ollama_host!r}")

        if self.config.tts_backend == "piper" and not self.config.piper_cmd:
            raise OmniReelError("tts_backend='piper' requires piper_cmd with {text_file} and {output} tokens.")

        if self.config.duration_seconds <= 0.5 or self.config.duration_seconds > 30:
            raise ValueError("duration_seconds must be between 0.5 and 30 seconds.")

        if self.config.qa_static_assets:
            LOGGER.warning(
                "qa_static_assets=True: using the local procedural motion backend. "
                "This creates real visible motion without external model weights, but it is still a deterministic QA/demo backend."
            )
            return

        if not self.config.diffusion_model_path.exists():
            raise FileNotFoundError(
                f"Diffusion model path does not exist: {self.config.diffusion_model_path}. "
                "Download/copy weights locally first; this pipeline will not fetch them."
            )
        if not self.config.liveportrait_cmd.strip():
            raise OmniReelError("liveportrait_cmd is required unless qa_static_assets=True.")

    def purge_vram(self, label: str) -> None:
        """Aggressively release Python and CUDA memory between heavyweight steps."""
        LOGGER.info("Purging Python/CUDA memory after step: %s", label)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                LOGGER.info("CUDA memory cache purged successfully.")
        except Exception as exc:
            LOGGER.warning("VRAM purge encountered a non-fatal issue: %s", exc)

    def run(self, topic: str) -> AssetManifest:
        """Generate all local media assets required by the Rust compositor."""
        started = time.perf_counter()
        LOGGER.info("Starting OmniReel local asset generation for topic=%r", topic)

        scene_plan = self.generate_scene_plan(topic)
        plan_path = self._write_scene_plan(scene_plan)
        self.purge_vram("ollama_scene_plan")

        base_image = self.generate_base_image(scene_plan)
        self.purge_vram("base_image")

        audio_path = self.synthesize_dialogue(scene_plan.dialogue)
        self.purge_vram("tts_audio")

        raw_video = self.animate(base_image, audio_path, scene_plan)
        self.purge_vram("local_animation")

        subtitles_json = self.generate_subtitle_timings(scene_plan.dialogue, audio_path)
        manifest = AssetManifest(
            output_dir=self.config.output_dir,
            plan_json=plan_path,
            base_image=base_image,
            audio=audio_path,
            raw_video=raw_video,
            subtitles_json=subtitles_json,
            final_video=self.config.output_dir / "final_omnireel.mp4",
            font_path=self.config.font_path,
            fps=self.config.fps,
        )
        manifest_path = self.config.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_json_dict(), indent=2), encoding="utf-8")
        LOGGER.info("Asset generation completed in %.2fs. Manifest: %s", time.perf_counter() - started, manifest_path)
        return manifest

    def generate_scene_plan(self, topic: str) -> ScenePlan:
        """Ask a local Ollama model for a strict JSON scene plan."""
        if not topic.strip():
            raise ValueError("topic must not be empty.")

        import ollama

        dialogue_budget = max(6, min(16, int(self.config.duration_seconds * 2.7)))
        system_prompt = (
            "You are OmniReel AI, a local video planning engine. Return compact valid JSON only. "
            "No markdown, no commentary, no reasoning. Required keys: "
            "title, visual_prompt, negative_prompt, dialogue, style_notes. "
            f"Dialogue must be under {dialogue_budget} words and fit a {self.config.duration_seconds:.1f} second video."
        )
        user_prompt = (
            f"Topic: {topic.strip()}\n"
            "Create one concrete visual scene with visible movement. "
            "For gravity, prefer a falling apple or ball. For space, prefer orbiting planets. "
            "Keep the narration extremely short."
        )

        LOGGER.info("Calling local Ollama model=%s for scene planning", self.config.ollama_model)
        client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
        try:
            response = client.chat(
                model=self.config.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": 0.1, "num_ctx": 1024, "num_predict": 180},
            )
        except Exception as exc:
            raise OmniReelError(
                "Local Ollama call failed. Ensure Ollama is installed, running locally, "
                f"and the model is pulled: {self.config.ollama_model}"
            ) from exc

        content = _extract_ollama_content(response)
        payload = _parse_json_object(content)
        return _validate_scene_plan(payload)

    def _write_scene_plan(self, scene_plan: ScenePlan) -> Path:
        plan_path = self.config.output_dir / "scene_plan.json"
        plan_path.write_text(json.dumps(asdict(scene_plan), indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Scene plan written: %s", plan_path)
        return plan_path

    def generate_base_image(self, scene_plan: ScenePlan) -> Path:
        """Generate a static base image from local Diffusers weights or procedural backend."""
        if self.config.qa_static_assets:
            return self._generate_procedural_preview_image(scene_plan)

        LOGGER.info("Loading local Diffusers pipeline from %s", self.config.diffusion_model_path)
        import torch
        from diffusers import DiffusionPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = _resolve_torch_dtype(torch, self.config.optimization, device)
        kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "local_files_only": True}

        if self.config.optimization == "int8":
            kwargs.update(_build_int8_kwargs())

        pipe: Optional[Any] = None
        try:
            pipe = DiffusionPipeline.from_pretrained(str(self.config.diffusion_model_path), **kwargs)
            pipe.set_progress_bar_config(disable=False)

            if device == "cuda":
                try:
                    pipe.enable_model_cpu_offload()
                    LOGGER.info("Enabled Diffusers CPU offload for lower peak VRAM.")
                except Exception:
                    pipe.to(device)
                    LOGGER.info("CPU offload unavailable; moved pipeline directly to CUDA.")

                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_tiling()
                    LOGGER.info("Enabled attention slicing and VAE tiling.")
                except Exception as exc:
                    LOGGER.debug("Memory-saver feature not available on this pipeline: %s", exc)
            else:
                pipe.to(device)

            generator = None
            if self.config.seed is not None:
                generator = torch.Generator(device=device).manual_seed(self.config.seed)

            LOGGER.info("Generating base image at %sx%s", self.config.width, self.config.height)
            result = pipe(
                prompt=scene_plan.visual_prompt,
                negative_prompt=scene_plan.negative_prompt or None,
                width=self.config.width,
                height=self.config.height,
                num_inference_steps=self.config.inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
            )
            if not getattr(result, "images", None):
                raise OmniReelError("Diffusers returned no images.")

            output_path = self.config.output_dir / "base_image.png"
            result.images[0].save(output_path)
            LOGGER.info("Base image written: %s", output_path)
            return output_path
        except Exception as exc:
            raise OmniReelError("Local Diffusers image generation failed.") from exc
        finally:
            if pipe is not None:
                del pipe
            self.purge_vram("diffusers_cleanup")

    def _generate_procedural_preview_image(self, scene_plan: ScenePlan) -> Path:
        from PIL import ImageFont

        output_path = self.config.output_dir / "base_image.png"
        image = self._draw_procedural_frame(scene_plan, progress=0.0, image_font_module=ImageFont)
        image.save(output_path)
        LOGGER.info("Procedural motion preview image written: %s", output_path)
        return output_path

    def synthesize_dialogue(self, dialogue: str) -> Path:
        """Synthesize narration with an offline TTS backend."""
        clean_dialogue = _normalize_text(dialogue)
        if not clean_dialogue:
            raise ValueError("dialogue must not be empty after normalization.")

        output_path = self.config.output_dir / "dialogue.wav"
        LOGGER.info("Synthesizing offline narration via backend=%s", self.config.tts_backend)
        if self.config.tts_backend == "pyttsx3":
            self._synthesize_with_pyttsx3(clean_dialogue, output_path)
        elif self.config.tts_backend == "piper":
            self._synthesize_with_piper(clean_dialogue, output_path)
        else:
            raise OmniReelError(f"Unsupported local TTS backend: {self.config.tts_backend}")

        _assert_file(output_path, "TTS audio")
        return output_path

    def _synthesize_with_pyttsx3(self, text: str, output_path: Path) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.setProperty("volume", 1.0)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            raise OmniReelError(
                "pyttsx3 offline TTS failed. On Linux, install espeak-ng; on Windows/macOS, "
                "ensure the system speech engine is available."
            ) from exc

    def _synthesize_with_piper(self, text: str, output_path: Path) -> None:
        if not self.config.piper_cmd:
            raise OmniReelError("piper_cmd is required for Piper TTS.")

        text_file = self.config.output_dir / "dialogue.txt"
        text_file.write_text(text, encoding="utf-8")
        command = _format_command(self.config.piper_cmd, {"text_file": text_file, "output": output_path})
        _run_local_command(command, "Piper offline TTS")

    def animate(self, image_path: Path, audio_path: Path, scene_plan: ScenePlan) -> Path:
        """Run local animation, external animation, or procedural motion."""
        _assert_file(image_path, "base image")
        _assert_file(audio_path, "dialogue audio")
        raw_video = self.config.output_dir / "raw_animated.mp4"

        if self.config.qa_static_assets:
            LOGGER.info("Running procedural local motion backend for %.2fs clip.", self.config.duration_seconds)
            self._render_procedural_motion_video(scene_plan, audio_path, raw_video)
        else:
            command = _format_command(
                self.config.liveportrait_cmd,
                {"image": image_path, "audio": audio_path, "output": raw_video},
            )
            LOGGER.info("Running local animation command.")
            _run_local_command(command, "LivePortrait/local animation")

        _assert_file(raw_video, "raw animated video")
        return raw_video

    def _render_procedural_motion_video(self, scene_plan: ScenePlan, audio_path: Path, output_path: Path) -> None:
        from PIL import ImageFont

        fps = max(1.0, float(self.config.fps))
        duration = max(0.5, float(self.config.duration_seconds))
        total_frames = max(2, int(round(fps * duration)))

        with tempfile.TemporaryDirectory(prefix="omnireel_frames_") as tmp:
            frame_dir = Path(tmp)
            for index in range(total_frames):
                progress = index / max(1, total_frames - 1)
                frame = self._draw_procedural_frame(scene_plan, progress=progress, image_font_module=ImageFont)
                frame.save(frame_dir / f"frame_{index + 1:05d}.png")

            frame_pattern = frame_dir / "frame_%05d.png"
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                _format_number(fps),
                "-i",
                str(frame_pattern),
                "-i",
                str(audio_path),
                "-t",
                _format_number(duration),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                str(output_path),
            ]
            _run_local_command(command, "procedural local motion render")

    def _draw_procedural_frame(self, scene_plan: ScenePlan, progress: float, image_font_module: Any) -> Any:
        from PIL import Image, ImageDraw

        width = max(320, int(self.config.width))
        height = max(320, int(self.config.height))
        width += width % 2
        height += height % 2
        progress = min(1.0, max(0.0, progress))

        theme = f"{scene_plan.title} {scene_plan.visual_prompt} {scene_plan.dialogue}".lower()
        image = Image.new("RGB", (width, height), color=(18, 24, 34))
        draw = ImageDraw.Draw(image)

        if any(word in theme for word in ["gravity", "apple", "fall", "falling", "pull"]):
            self._draw_gravity_scene(draw, image_font_module, scene_plan, width, height, progress)
        elif any(word in theme for word in ["orbit", "planet", "sun", "space"]):
            self._draw_orbit_scene(draw, image_font_module, scene_plan, width, height, progress)
        else:
            self._draw_generic_motion_scene(draw, image_font_module, scene_plan, width, height, progress)

        return image

    def _draw_gravity_scene(self, draw: Any, image_font_module: Any, scene_plan: ScenePlan, width: int, height: int, progress: float) -> None:
        title_font = _load_pillow_font(image_font_module, self.config.font_path, max(26, width // 22))
        small_font = _load_pillow_font(image_font_module, self.config.font_path, max(16, width // 46))

        # Classroom background with parallax-like motion lines.
        draw.rectangle([(0, 0), (width, height)], fill=(21, 29, 38))
        draw.rectangle([(0, int(height * 0.72)), (width, height)], fill=(54, 39, 28))
        for x in range(-width, width * 2, max(80, width // 8)):
            x_shift = int((progress * width * 0.15) % max(80, width // 8))
            draw.line([(x + x_shift, int(height * 0.72)), (x + x_shift + width // 5, height)], fill=(75, 55, 40), width=2)

        board = (int(width * 0.08), int(height * 0.10), int(width * 0.92), int(height * 0.34))
        draw.rounded_rectangle(board, radius=18, fill=(19, 74, 62), outline=(180, 215, 190), width=3)
        title = scene_plan.title or "Gravity"
        for line_index, line in enumerate(_wrap_for_pillow(draw, title, title_font, int(width * 0.74))[:2]):
            draw.text((board[0] + 28, board[1] + 24 + line_index * (title_font.size + 8)), line, font=title_font, fill=(245, 255, 235))

        # Simple teacher/kid figure.
        cx = int(width * 0.22)
        body_y = int(height * 0.62)
        draw.ellipse([(cx - 42, body_y - 180), (cx + 42, body_y - 96)], fill=(178, 122, 75), outline=(40, 30, 25), width=2)
        draw.rectangle([(cx - 46, body_y - 96), (cx + 46, body_y + 60)], fill=(70, 105, 180), outline=(30, 50, 100), width=2)
        arm_angle = math.sin(progress * math.pi * 2) * 0.25
        draw.line([(cx + 36, body_y - 55), (int(cx + width * 0.18), int(body_y - 120 + 40 * arm_angle))], fill=(178, 122, 75), width=12)

        # Apple falling with acceleration, trail, and bounce hint.
        top_y = int(height * 0.23)
        bottom_y = int(height * 0.70)
        fall = progress * progress
        apple_x = int(width * 0.66 + math.sin(progress * math.pi * 2) * width * 0.03)
        apple_y = int(top_y + (bottom_y - top_y) * fall)
        for i in range(6):
            trail_p = max(0.0, progress - i * 0.055)
            trail_y = int(top_y + (bottom_y - top_y) * trail_p * trail_p)
            alpha_color = (180 - i * 18, 70, 52)
            r = max(5, int(width * 0.015) - i)
            draw.ellipse([(apple_x - r, trail_y - r), (apple_x + r, trail_y + r)], fill=alpha_color)

        apple_r = max(20, width // 26)
        draw.ellipse([(apple_x - apple_r, apple_y - apple_r), (apple_x + apple_r, apple_y + apple_r)], fill=(210, 48, 38), outline=(95, 18, 18), width=3)
        draw.ellipse([(apple_x - apple_r // 3, apple_y - apple_r // 2), (apple_x, apple_y - apple_r // 4)], fill=(245, 125, 105))
        draw.line([(apple_x, apple_y - apple_r), (apple_x + apple_r // 3, apple_y - apple_r - apple_r // 2)], fill=(90, 52, 20), width=4)
        draw.ellipse([(apple_x + apple_r // 4, apple_y - apple_r - apple_r // 2), (apple_x + apple_r, apple_y - apple_r // 2)], fill=(80, 160, 70))

        arrow_x = int(width * 0.82)
        draw.line([(arrow_x, top_y), (arrow_x, bottom_y)], fill=(255, 210, 90), width=6)
        draw.polygon([(arrow_x - 18, bottom_y - 20), (arrow_x + 18, bottom_y - 20), (arrow_x, bottom_y + 20)], fill=(255, 210, 90))
        draw.text((arrow_x - 64, bottom_y + 28), "pulls down", font=small_font, fill=(255, 225, 130))

    def _draw_orbit_scene(self, draw: Any, image_font_module: Any, scene_plan: ScenePlan, width: int, height: int, progress: float) -> None:
        title_font = _load_pillow_font(image_font_module, self.config.font_path, max(26, width // 24))
        draw.rectangle([(0, 0), (width, height)], fill=(5, 8, 20))
        for i in range(80):
            x = (i * 97 + int(progress * 120)) % width
            y = (i * 173) % height
            draw.ellipse([(x, y), (x + 2, y + 2)], fill=(210, 220, 255))
        cx, cy = width // 2, int(height * 0.48)
        sun_r = max(36, width // 11)
        draw.ellipse([(cx - sun_r, cy - sun_r), (cx + sun_r, cy + sun_r)], fill=(255, 190, 50), outline=(255, 230, 100), width=4)
        for orbit_r, color, phase in [(int(width * 0.27), (90, 150, 255), 0.0), (int(width * 0.38), (110, 220, 160), 1.8)]:
            draw.ellipse([(cx - orbit_r, cy - orbit_r), (cx + orbit_r, cy + orbit_r)], outline=(80, 95, 130), width=2)
            angle = progress * math.tau + phase
            px = int(cx + math.cos(angle) * orbit_r)
            py = int(cy + math.sin(angle) * orbit_r)
            pr = max(16, width // 32)
            draw.ellipse([(px - pr, py - pr), (px + pr, py + pr)], fill=color, outline=(230, 240, 255), width=2)
        for line_index, line in enumerate(_wrap_for_pillow(draw, scene_plan.title, title_font, int(width * 0.84))[:2]):
            draw.text((int(width * 0.08), int(height * 0.07) + line_index * (title_font.size + 6)), line, font=title_font, fill=(245, 248, 255))

    def _draw_generic_motion_scene(self, draw: Any, image_font_module: Any, scene_plan: ScenePlan, width: int, height: int, progress: float) -> None:
        title_font = _load_pillow_font(image_font_module, self.config.font_path, max(26, width // 23))
        draw.rectangle([(0, 0), (width, height)], fill=(22, 30, 48))
        for i in range(10):
            x = int((i * width * 0.16 + progress * width * 0.28) % width)
            y = int(height * (0.20 + 0.05 * math.sin(i)))
            draw.ellipse([(x - 14, y - 14), (x + 14, y + 14)], fill=(70, 120, 220))
        for line_index, line in enumerate(_wrap_for_pillow(draw, scene_plan.title, title_font, int(width * 0.84))[:2]):
            draw.text((int(width * 0.08), int(height * 0.08) + line_index * (title_font.size + 8)), line, font=title_font, fill=(255, 255, 255))
        ball_x = int(width * (0.12 + 0.76 * progress))
        ball_y = int(height * 0.58 + math.sin(progress * math.tau * 2) * height * 0.08)
        radius = max(32, width // 14)
        draw.ellipse([(ball_x - radius, ball_y - radius), (ball_x + radius, ball_y + radius)], fill=(240, 120, 70), outline=(255, 225, 190), width=4)
        draw.line([(int(width * 0.10), int(height * 0.72)), (int(width * 0.90), int(height * 0.72))], fill=(100, 160, 255), width=5)

    def generate_subtitle_timings(self, dialogue: str, audio_path: Path) -> Path:
        """Generate Whisper-compatible subtitle timing JSON locally."""
        _assert_file(audio_path, "dialogue audio")
        output_path = self.config.output_dir / "whisper_timestamps.json"

        if self.config.whisper_cmd:
            json_no_ext = output_path.with_suffix("")
            command = _format_command(
                self.config.whisper_cmd,
                {"audio": audio_path, "json": output_path, "json_no_ext": json_no_ext},
            )
            LOGGER.info("Running local Whisper command for subtitle timestamps.")
            _run_local_command(command, "Whisper timestamp generation")
            _assert_file(output_path, "Whisper timestamp JSON")
            return output_path

        LOGGER.warning(
            "No whisper_cmd configured. Writing deterministic heuristic timings; "
            "for production speech-accurate captions, pass a local whisper.cpp/faster-whisper command."
        )
        payload = _build_heuristic_whisper_json(dialogue, max_duration_seconds=self.config.duration_seconds)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path


def _is_loopback_url(url: str) -> bool:
    return bool(re.match(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?/?", url))


def _extract_ollama_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message", {})
        return str(message.get("content", ""))

    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", ""))
    if message is not None and hasattr(message, "content"):
        return str(message.content)

    raise OmniReelError("Could not extract message content from Ollama response.")


def _parse_json_object(content: str) -> Dict[str, Any]:
    stripped = content.strip().lstrip("\ufeff")
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise OmniReelError(f"LLM output was not valid JSON: {content[:500]!r}") from exc
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise OmniReelError("LLM JSON output must be an object.")
    return payload


def _validate_scene_plan(payload: Dict[str, Any]) -> ScenePlan:
    required = ["title", "visual_prompt", "negative_prompt", "dialogue", "style_notes"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise OmniReelError(f"LLM JSON missing keys: {missing}")

    plan = ScenePlan(
        title=_normalize_text(str(payload["title"]))[:120],
        visual_prompt=_normalize_text(str(payload["visual_prompt"])),
        negative_prompt=_normalize_text(str(payload.get("negative_prompt", "low quality, blurry, distorted"))),
        dialogue=_normalize_text(str(payload["dialogue"])),
        style_notes=_normalize_text(str(payload["style_notes"])),
    )
    if len(plan.visual_prompt) < 20:
        raise OmniReelError("visual_prompt is too short for reliable image generation.")
    if len(plan.dialogue.split()) < 3:
        raise OmniReelError("dialogue is too short for narration.")
    return plan


def _resolve_torch_dtype(torch_module: Any, optimization: DiffusionOptimization, device: str) -> Any:
    if optimization == "fp32" or device == "cpu":
        return torch_module.float32
    if optimization == "bf16":
        return torch_module.bfloat16
    if optimization in {"auto", "fp16", "int8"}:
        return torch_module.float16
    raise ValueError(f"Unsupported optimization mode: {optimization}")


def _build_int8_kwargs() -> Dict[str, Any]:
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise OmniReelError(
            "optimization='int8' requires a transformers build with BitsAndBytesConfig "
            "and a compatible bitsandbytes installation."
        ) from exc

    return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "device_map": "balanced"}


def _load_pillow_font(image_font_module: Any, font_path: Optional[Path], size: int) -> Any:
    candidates = []
    if font_path:
        candidates.append(font_path)
    if os.name == "nt":
        candidates.extend([Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")])
    candidates.extend([
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ])

    for candidate in candidates:
        try:
            if candidate.exists():
                return image_font_module.truetype(str(candidate), size=size)
        except Exception:
            continue
    return image_font_module.load_default()


def _font_line_height(font: Any) -> int:
    try:
        bbox = font.getbbox("Ag")
        return int((bbox[3] - bbox[1]) * 1.25)
    except Exception:
        return 24


def _wrap_for_pillow(draw: Any, text: str, font: Any, max_width: int) -> List[str]:
    words = _normalize_text(text).split()
    if not words:
        return []

    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _assert_file(path: Path, label: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Expected {label} file was not created or is empty: {path}")


def _format_command(template: str, values: Dict[str, Path]) -> List[str]:
    if not template.strip():
        raise ValueError("Command template must not be empty.")

    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))

    unresolved = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", rendered)
    if unresolved:
        raise ValueError(f"Unresolved command template tokens: {unresolved}")

    return shlex.split(rendered, posix=os.name != "nt")


def _run_local_command(command: Sequence[str], label: str) -> None:
    LOGGER.info("Executing %s: %s", label, " ".join(shlex.quote(part) for part in command))
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ExternalCommandError(f"{label} executable not found: {command[0]}") from exc

    if completed.returncode != 0:
        raise ExternalCommandError(
            f"{label} failed with exit code {completed.returncode}.\n"
            f"STDOUT:\n{completed.stdout[-4000:]}\n"
            f"STDERR:\n{completed.stderr[-4000:]}"
        )

    if completed.stdout.strip():
        LOGGER.debug("%s stdout: %s", label, completed.stdout[-2000:])
    if completed.stderr.strip():
        LOGGER.debug("%s stderr: %s", label, completed.stderr[-2000:])


def _build_heuristic_whisper_json(dialogue: str, max_duration_seconds: Optional[float] = None) -> Dict[str, Any]:
    words = re.findall(r"[\w'’-]+|[^\w\s]", _normalize_text(dialogue), flags=re.UNICODE)
    spoken_words = [word for word in words if re.search(r"\w", word)]
    if not spoken_words:
        raise ValueError("Cannot build subtitle timings from empty dialogue.")

    pause_every = 5 if max_duration_seconds and max_duration_seconds <= 4 else 8
    timed_words: List[Dict[str, Any]] = []
    if max_duration_seconds:
        usable = max(0.5, max_duration_seconds * 0.92)
        slot = usable / max(1, len(spoken_words))
        for idx, word in enumerate(spoken_words):
            start = idx * slot
            end = min(usable, start + slot * 0.82)
            timed_words.append({"word": word, "start": round(start, 3), "end": round(end, 3), "probability": 1.0})
    else:
        seconds_per_word = 0.38
        current = 0.0
        for idx, word in enumerate(spoken_words):
            start = current
            end = start + seconds_per_word
            timed_words.append({"word": word, "start": round(start, 3), "end": round(end, 3), "probability": 1.0})
            current = end + (0.18 if (idx + 1) % pause_every == 0 else 0.04)

    segments: List[Dict[str, Any]] = []
    for idx in range(0, len(timed_words), pause_every):
        chunk = timed_words[idx : idx + pause_every]
        text = " ".join(item["word"] for item in chunk)
        segments.append(
            {
                "id": idx // pause_every,
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
                "text": text,
                "words": chunk,
            }
        )

    return {"text": _normalize_text(dialogue), "segments": segments, "language": "en", "source": "heuristic-local"}


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")
