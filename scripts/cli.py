from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from .models_pipeline import PipelineConfig
from .orchestrator import OmniReelOrchestrator, OrchestratorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnireel-ai")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output-dir", default="outputs/run", type=Path)
    parser.add_argument("--ollama-model", default="llama3.1:8b")
    parser.add_argument("--diffusion-model", type=Path, default=None)
    parser.add_argument("--liveportrait-cmd", default=None)
    parser.add_argument("--whisper-cmd", default=None)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument("--tts-backend", choices=["pyttsx3", "piper"], default="pyttsx3")
    parser.add_argument("--piper-cmd", default=None)
    parser.add_argument("--optimization", choices=["auto", "fp16", "bf16", "fp32", "int8"], default="auto")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--qa-static-assets", action="store_true")
    parser.add_argument("--no-llm-planning", action="store_true")
    parser.add_argument("--video-backend", choices=["liveportrait", "procedural", "comfyui"], default=None)
    parser.add_argument("--comfyui-workflow", type=Path, default=None)
    parser.add_argument("--comfyui-server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfyui-timeout", type=float, default=3600.0)
    parser.add_argument("--comfyui-poll-interval", type=float, default=5.0)
    parser.add_argument("--comfyui-prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--comfyui-set", dest="comfyui_set_values", action="append", default=[])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rust-binary", type=Path, default=None)
    parser.add_argument("--debug-rust", action="store_true")
    parser.add_argument("--skip-python-generation", action="store_true")
    parser.add_argument("--existing-manifest", type=Path, default=None)
    parser.add_argument("--rust-threads", type=int, default=None)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--audio-bitrate", default="192k")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    video_backend = args.video_backend or ("procedural" if args.qa_static_assets else "liveportrait")
    if video_backend == "procedural":
        args.qa_static_assets = True

    if args.skip_python_generation:
        if args.existing_manifest is None:
            parser.error("--existing-manifest is required when --skip-python-generation is used")
        if video_backend == "comfyui":
            parser.error("--skip-python-generation cannot be combined with --video-backend comfyui")
    elif video_backend == "comfyui":
        if args.comfyui_workflow is None:
            parser.error("--comfyui-workflow is required when --video-backend comfyui is used")
    elif not args.qa_static_assets:
        if args.diffusion_model is None:
            parser.error("--diffusion-model is required unless --skip-python-generation, --qa-static-assets, or --video-backend comfyui is used")
        if not args.liveportrait_cmd:
            parser.error("--liveportrait-cmd is required unless --skip-python-generation, --qa-static-assets, or --video-backend comfyui is used")

    try:
        if video_backend == "comfyui":
            from .run_comfyui_omnireel import CHILD_SAFE_NEGATIVE_PROMPT, run_comfyui_omnireel

            manifest = run_comfyui_omnireel(
                SimpleNamespace(
                    topic=args.topic,
                    output_dir=args.output_dir,
                    workflow=args.comfyui_workflow,
                    server=args.comfyui_server,
                    ollama_model=args.ollama_model,
                    no_llm_planning=args.no_llm_planning,
                    prompt=args.comfyui_prompt,
                    negative_prompt=args.negative_prompt or CHILD_SAFE_NEGATIVE_PROMPT,
                    font=args.font,
                    tts_backend=args.tts_backend,
                    piper_cmd=args.piper_cmd,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    duration_seconds=args.duration_seconds,
                    seed=args.seed,
                    timeout=args.comfyui_timeout,
                    poll_interval=args.comfyui_poll_interval,
                    set_values=args.comfyui_set_values,
                    project_root=args.project_root,
                    rust_binary=args.rust_binary,
                    rust_threads=args.rust_threads,
                    crf=args.crf,
                    audio_bitrate=args.audio_bitrate,
                )
            )
            logging.getLogger(__name__).info("OmniReel complete: %s", manifest.final_video)
            return 0

        pipeline_config = PipelineConfig(
            output_dir=args.output_dir,
            ollama_model=args.ollama_model,
            diffusion_model_path=args.diffusion_model or Path("."),
            liveportrait_cmd=args.liveportrait_cmd or "",
            whisper_cmd=args.whisper_cmd,
            font_path=args.font,
            tts_backend=args.tts_backend,
            piper_cmd=args.piper_cmd,
            optimization=args.optimization,
            width=args.width,
            height=args.height,
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            fps=args.fps,
            duration_seconds=args.duration_seconds,
            allow_remote_backends=False,
            qa_static_assets=args.qa_static_assets,
            no_llm_planning=args.no_llm_planning,
        )
        orchestrator_config = OrchestratorConfig(
            pipeline=pipeline_config,
            project_root=args.project_root,
            rust_binary=args.rust_binary,
            build_release=not args.debug_rust,
            skip_python_generation=args.skip_python_generation,
            existing_manifest=args.existing_manifest,
            rust_threads=args.rust_threads,
            rust_crf=args.crf,
            audio_bitrate=args.audio_bitrate,
        )
        manifest = OmniReelOrchestrator(orchestrator_config).run(args.topic)
        logging.getLogger(__name__).info("OmniReel complete: %s", manifest.final_video)
        return 0
    except Exception as exc:
        logging.getLogger(__name__).exception("OmniReel failed: %s", exc)
        return 1
