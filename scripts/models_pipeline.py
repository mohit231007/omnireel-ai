"""Local AI asset generation for OmniReel AI.

Python owns model orchestration only:
- Local LLM planning through Ollama.
- Local Diffusers image generation from an already-downloaded model path.
- Offline TTS through pyttsx3 or a user-provided Piper command.
- Local animation through a user-provided LivePortrait-compatible command.
- Optional local Whisper/whisper.cpp timestamp extraction.

No remote APIs are called by this module. Any backend that requires a network service
is either rejected or treated as a locality violation.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shlex
import subprocess
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
    allow_remote_backends: bool = False


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

        if not self.config.diffusion_model_path.exists():
            raise FileNotFoundError(
                f"Diffusion model path does not exist: {self.config.diffusion_model_path}. "
                "Download/copy weights locally first; this pipeline will not fetch them."
            )

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
        self.purge_vram("diffusers_base_image")

        audio_path = self.synthesize_dialogue(scene_plan.dialogue)
        self.purge_vram("tts_audio")

        raw_video = self.animate(base_image, audio_path)
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

        system_prompt = (
            "You are OmniReel AI, a local video planning engine. "
            "Return only valid JSON with keys: title, visual_prompt, negative_prompt, dialogue, style_notes. "
            "No markdown, no commentary, no code fences. Keep dialogue under 80 words."
        )
        user_prompt = (
            f"Create a short educational/social video scene for this topic: {topic.strip()}\n"
            "The visual prompt must describe one clear character/scene suitable for image generation. "
            "The dialogue must be natural spoken narration."
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
                options={"temperature": 0.45, "num_ctx": 4096},
            )
        except Exception as exc:
            raise OmniReelError(
                "Local Ollama call failed. Ensure Ollama is installed, running locally, "
                f"and the model is pulled: {self.config.ollama_model}"
            ) from exc

        content = response.get("message", {}).get("content", "") if isinstance(response, dict) else ""
        payload = _parse_json_object(content)
        return _validate_scene_plan(payload)

    def _write_scene_plan(self, scene_plan: ScenePlan) -> Path:
        plan_path = self.config.output_dir / "scene_plan.json"
        plan_path.write_text(json.dumps(asdict(scene_plan), indent=2, ensure_ascii=False), encoding="utf-8")
        LOGGER.info("Scene plan written: %s", plan_path)
        return plan_path

    def generate_base_image(self, scene_plan: ScenePlan) -> Path:
        """Generate a static base image from local Diffusers weights."""
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
            engine.setProperty("rate", 165)
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

    def animate(self, image_path: Path, audio_path: Path) -> Path:
        """Run a local LivePortrait-compatible CLI to generate a raw talking-head video."""
        _assert_file(image_path, "base image")
        _assert_file(audio_path, "dialogue audio")
        raw_video = self.config.output_dir / "raw_animated.mp4"
        command = _format_command(
            self.config.liveportrait_cmd,
            {"image": image_path, "audio": audio_path, "output": raw_video},
        )
        LOGGER.info("Running local animation command.")
        _run_local_command(command, "LivePortrait/local animation")
        _assert_file(raw_video, "raw animated video")
        return raw_video

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
        payload = _build_heuristic_whisper_json(dialogue)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path


def _is_loopback_url(url: str) -> bool:
    return bool(re.match(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?/?", url))


def _parse_json_object(content: str) -> Dict[str, Any]:
    stripped = content.strip()
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


def _build_heuristic_whisper_json(dialogue: str) -> Dict[str, Any]:
    words = re.findall(r"[\w'’-]+|[^\w\s]", _normalize_text(dialogue), flags=re.UNICODE)
    spoken_words = [word for word in words if re.search(r"\w", word)]
    if not spoken_words:
        raise ValueError("Cannot build subtitle timings from empty dialogue.")

    seconds_per_word = 0.38
    pause_every = 8
    current = 0.0
    timed_words: List[Dict[str, Any]] = []
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
