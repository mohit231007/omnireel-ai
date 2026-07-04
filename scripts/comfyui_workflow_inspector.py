"""Inspect ComfyUI API workflow JSON for OmniReel integration.

Use this after exporting a ComfyUI workflow in API format. It prints likely prompt,
image, video-save, and model nodes, plus ready-to-copy --set paths for the
ComfyUI adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROMPT_KEYS = {"text", "prompt", "positive", "negative", "conditioning"}
IMAGE_KEYS = {"image", "input_image", "init_image"}
VIDEO_KEYS = {"filename_prefix", "format", "codec", "fps", "frame_rate", "save_output"}
MODEL_KEYS = {"ckpt_name", "model_name", "unet_name", "clip_name", "vae_name"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comfyui-workflow-inspector")
    parser.add_argument("workflow", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workflow = json.loads(args.workflow.read_text(encoding="utf-8-sig"))
    if not isinstance(workflow, dict):
        raise ValueError("Expected ComfyUI API workflow JSON object.")

    print(f"Workflow: {args.workflow}")
    print(f"Nodes: {len(workflow)}")
    print()

    prompt_nodes: List[Tuple[str, str, Dict[str, Any]]] = []
    image_nodes: List[Tuple[str, str, Dict[str, Any]]] = []
    video_nodes: List[Tuple[str, str, Dict[str, Any]]] = []
    model_nodes: List[Tuple[str, str, Dict[str, Any]]] = []

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        keys = set(inputs.keys())
        class_lower = class_type.lower()
        text_like = any(isinstance(value, str) and len(value.strip()) > 12 for value in inputs.values())

        if keys & PROMPT_KEYS or "text" in class_lower or text_like:
            prompt_nodes.append((node_id, class_type, inputs))
        if keys & IMAGE_KEYS or "image" in class_lower:
            image_nodes.append((node_id, class_type, inputs))
        if keys & VIDEO_KEYS or "video" in class_lower or "vhs" in class_lower:
            video_nodes.append((node_id, class_type, inputs))
        if keys & MODEL_KEYS or any(part in class_lower for part in ["checkpoint", "model", "clip", "vae"]):
            model_nodes.append((node_id, class_type, inputs))

    _print_section("Prompt/Text candidates", prompt_nodes, ["text", "prompt", "positive", "negative"])
    _print_section("Image/Input candidates", image_nodes, ["image", "input_image", "init_image"])
    _print_section("Video/Save candidates", video_nodes, ["filename_prefix", "format", "codec", "fps", "frame_rate"])
    _print_section("Model candidates", model_nodes, ["ckpt_name", "model_name", "unet_name", "clip_name", "vae_name"])

    print("Suggested adapter overrides:")
    for node_id, class_type, inputs in prompt_nodes:
        if "text" in inputs:
            current = str(inputs.get("text", "")).lower()
            token = "{negative_prompt}" if any(word in current for word in ["bad", "blurry", "ugly", "negative"]) else "{prompt}"
            print(f'  --set "{node_id}.inputs.text={token}"    # {class_type}')
    for node_id, class_type, inputs in image_nodes:
        if "image" in inputs:
            print(f'  --set "{node_id}.inputs.image={{input_image}}"    # {class_type}')
    for node_id, class_type, inputs in video_nodes:
        if "filename_prefix" in inputs:
            print(f'  --set "{node_id}.inputs.filename_prefix={{filename_prefix}}"    # {class_type}')
        if "fps" in inputs:
            print(f'  --set "{node_id}.inputs.fps={{fps}}"    # {class_type}')

    return 0


def _print_section(title: str, rows: Iterable[Tuple[str, str, Dict[str, Any]]], keys: List[str]) -> None:
    rows = list(rows)
    print(f"{title} ({len(rows)})")
    print("-" * max(8, len(title)))
    for node_id, class_type, inputs in rows:
        print(f"Node {node_id}: {class_type}")
        for key in keys:
            if key in inputs:
                value = inputs[key]
                if isinstance(value, str):
                    value = value.replace("\n", " ")[:180]
                print(f"  {key}: {value}")
        text_values = [value for value in inputs.values() if isinstance(value, str) and len(value.strip()) > 30]
        for value in text_values[:2]:
            print(f"  text-like: {value.replace(chr(10), ' ')[:180]}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
