"""End-to-end OmniReel pipeline using a real local ComfyUI video backend.

This is the production-direction path:

1. Build a short educational scene plan locally.
2. Create a local base image for image-to-video conditioning.
3. Generate offline narration.
4. Render real model video through a loopback-only ComfyUI workflow.
5. Generate local subtitle timings.
6. Hand off to the Rust compositor for final subtitle burn + audio mux.

The script is intentionally separate from the default run.py while the real video
backend is being stabilized.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .comfyui_video_client import render_with_comfyui
from .models_pipeline import AssetManifest, OmniReelPipeline, PipelineConfig
from .orchestrator import OmniReelOrchestrator, OrchestratorConfig

LOGGER = logging.getLogger(__name__)

DEFAULT_LTX_API_SETS = (
    "6.inputs.text={prompt}",
    "7.inputs.text={negative_prompt}",
    "78.inputs.image={input_image}",
    "80.inputs.fps={fps}",
    "81.inputs.filename_prefix={filename_prefix}",
)

CHILD_SAFE_NEGATIVE_PROMPT = (
    "adult, swimsuit, bikini, lingerie, cleavage, nude, sexual, suggestive, glamour, "
    "fashion model, realistic woman, body focus, unsafe for children, scary, violent, "
    "blurry, distorted, low quality"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnireel-comfyui")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comfyui_omnireel"))
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--ollama-model", default="qwen3:4b")
    parser.add_argument("--no-llm-planning", action="store_true", default=True)
    parser.add_argument("--use-llm-planning", action="store_false", dest="no_llm_planning")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=CHILD_SAFE_NEGATIVE_PROMPT)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument("--tts-backend", choices=["pyttsx3", "piper"], default="pyttsx3")
    parser.add_argument("--piper-cmd", default=None)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--set", dest="set_values", action="append", default=[])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rust-binary", type=Path, default=None)
    parser.add_argument("--rust-threads", type=int, default=None)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        final_manifest = run_comfyui_omnireel(args)
        LOGGER.info("OmniReel ComfyUI complete: %s", final_manifest.final_video)
        return 0
    except Exception as exc:
        LOGGER.exception("OmniReel ComfyUI failed: %s", exc)
        return 1


def run_comfyui_omnireel(args: argparse.Namespace) -> AssetManifest:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_config = PipelineConfig(
        output_dir=args.output_dir,
        ollama_model=args.ollama_model,
        diffusion_model_path=Path("."),
        liveportrait_cmd="",
        whisper_cmd=None,
        font_path=args.font,
        tts_backend=args.tts_backend,
        piper_cmd=args.piper_cmd,
        width=args.width,
        height=args.height,
        seed=args.seed,
        fps=args.fps,
        duration_seconds=args.duration_seconds,
        allow_remote_backends=False,
        qa_static_assets=True,
        no_llm_planning=args.no_llm_planning,
    )
    pipeline = OmniReelPipeline(pipeline_config)

    scene_plan = pipeline.generate_scene_plan(args.topic)
    plan_path = args.output_dir / "scene_plan.json"
    plan_path.write_text(json.dumps(asdict(scene_plan), indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Scene plan written: %s", plan_path)

    base_image = pipeline.generate_base_image(scene_plan)
    pipeline.purge_vram("base_image")

    audio_path = pipeline.synthesize_dialogue(scene_plan.dialogue)
    pipeline.purge_vram("tts_audio")

    raw_video = args.output_dir / "raw_model_video.mp4"
    prompt = args.prompt or _child_safe_positive_prompt(scene_plan.visual_prompt)
    set_values = list(DEFAULT_LTX_API_SETS) + list(args.set_values or [])

    LOGGER.info("Rendering real model video through local ComfyUI workflow: %s", args.workflow)
    render_with_comfyui(
        server=args.server,
        workflow_path=args.workflow,
        output_path=raw_video,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        input_image=base_image,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        set_values=set_values,
    )
    pipeline.purge_vram("comfyui_real_model_video")

    subtitles_json = pipeline.generate_subtitle_timings(scene_plan.dialogue, audio_path)
    manifest = AssetManifest(
        output_dir=args.output_dir,
        plan_json=plan_path,
        base_image=base_image,
        audio=audio_path,
        raw_video=raw_video,
        subtitles_json=subtitles_json,
        final_video=args.output_dir / "final_omnireel.mp4",
        font_path=args.font,
        fps=args.fps,
    )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_json_dict(), indent=2), encoding="utf-8")
    LOGGER.info("Manifest written: %s", manifest_path)

    orchestrator_config = OrchestratorConfig(
        pipeline=pipeline_config,
        project_root=args.project_root,
        rust_binary=args.rust_binary,
        build_release=True,
        skip_python_generation=True,
        existing_manifest=manifest_path,
        rust_threads=args.rust_threads,
        rust_crf=args.crf,
        audio_bitrate=args.audio_bitrate,
    )
    return OmniReelOrchestrator(orchestrator_config).run(args.topic)


def _child_safe_positive_prompt(prompt: str) -> str:
    return (
        f"{prompt}, simple educational science animation, child-safe, clean classroom visual, "
        "no people, no swimsuit, no fashion pose, no body focus"
    )


if __name__ == "__main__":
    raise SystemExit(main())
