"""Local ComfyUI video backend adapter for OmniReel AI.

This script talks only to a loopback ComfyUI server and queues an API-format
workflow JSON. It is intentionally dependency-free and uses Python stdlib only.

The workflow can contain placeholders in any string value:

- {prompt}
- {negative_prompt}
- {input_image}
- {filename_prefix}
- {seed}
- {duration_seconds}
- {fps}
- {width}
- {height}

You can also patch explicit workflow fields with repeated --set values:

    --set "6.inputs.text={prompt}"
    --set "27.inputs.filename_prefix={filename_prefix}"

The script waits for ComfyUI to finish, finds the first video-like output in the
history payload, downloads it from /view, and writes it to --output.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
import uuid
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".gif"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comfyui-video-client")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--input-image", type=Path, default=None)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--set", dest="sets", action="append", default=[])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    render_with_comfyui(
        server=args.server,
        workflow_path=args.workflow,
        output_path=args.output,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=args.input_image,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        set_values=args.sets,
    )
    return 0


def render_with_comfyui(
    *,
    server: str,
    workflow_path: Path,
    output_path: Path,
    prompt: str,
    negative_prompt: str = "",
    input_image: Optional[Path] = None,
    duration_seconds: float = 3.0,
    fps: float = 30.0,
    width: int = 720,
    height: int = 1280,
    seed: int = 42,
    timeout: float = 1800.0,
    poll_interval: float = 2.0,
    set_values: Optional[Iterable[str]] = None,
) -> Path:
    server = _normalize_loopback_server(server)
    if not workflow_path.exists():
        raise FileNotFoundError(f"ComfyUI workflow not found: {workflow_path}")

    workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
    client_id = str(uuid.uuid4())
    filename_prefix = f"omnireel_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    uploaded_image_name = ""
    if input_image:
        if not input_image.exists():
            raise FileNotFoundError(f"ComfyUI input image not found: {input_image}")
        uploaded_image_name = upload_image(server, input_image)

    replacements = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "input_image": uploaded_image_name,
        "filename_prefix": filename_prefix,
        "seed": str(seed),
        "duration_seconds": _format_number(duration_seconds),
        "fps": _format_number(fps),
        "width": str(width),
        "height": str(height),
    }

    workflow = _replace_tokens(workflow, replacements)
    for raw_set in set_values or []:
        path, value = _parse_set(raw_set)
        patched_value = _replace_tokens(value, replacements)
        _apply_path(workflow, path, _coerce_value(patched_value))

    prompt_id = queue_prompt(server, workflow, client_id)
    history = wait_for_history(server, prompt_id, timeout=timeout, poll_interval=poll_interval)
    output_ref = find_first_video_output(history, prompt_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    download_output(server, output_ref, output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ComfyUI output was not created: {output_path}")
    return output_path


def upload_image(server: str, image_path: Path) -> str:
    boundary = f"----OmniReelBoundary{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    file_bytes = image_path.read_bytes()
    filename = image_path.name

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode())
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n')
    body.extend(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        _join_url(server, "/upload/image"),
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    payload = _json_request(request)
    return str(payload.get("name") or payload.get("filename") or filename)


def queue_prompt(server: str, workflow: Dict[str, Any], client_id: str) -> str:
    body = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
    request = urllib.request.Request(
        _join_url(server, "/prompt"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    payload = _json_request(request)
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI /prompt response did not contain prompt_id: {payload}")
    return str(prompt_id)


def wait_for_history(server: str, prompt_id: str, *, timeout: float, poll_interval: float) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            payload = _json_get(_join_url(server, f"/history/{prompt_id}"))
        except Exception:
            payload = {}
        if payload:
            record = payload.get(prompt_id, payload)
            if isinstance(record, dict) and record.get("outputs"):
                return payload
        time.sleep(max(0.25, poll_interval))
    raise TimeoutError(f"Timed out waiting for ComfyUI prompt_id={prompt_id}")


def find_first_video_output(history: Dict[str, Any], prompt_id: str) -> Dict[str, str]:
    record = history.get(prompt_id, history)
    outputs = record.get("outputs", {}) if isinstance(record, dict) else {}
    candidates: List[Dict[str, str]] = []
    image_candidates: List[Dict[str, str]] = []

    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in ("videos", "gifs", "images"):
            for item in node_output.get(key, []) or []:
                if not isinstance(item, dict):
                    continue
                filename = str(item.get("filename", ""))
                if not filename:
                    continue
                ref = {
                    "filename": filename,
                    "subfolder": str(item.get("subfolder", "")),
                    "type": str(item.get("type", "output")),
                }
                suffix = Path(filename).suffix.lower()
                if suffix in VIDEO_EXTENSIONS:
                    candidates.append(ref)
                elif suffix in IMAGE_EXTENSIONS:
                    image_candidates.append(ref)

    if candidates:
        return candidates[0]

    if image_candidates:
        raise RuntimeError(
            "ComfyUI workflow completed but returned image files only. Use a video workflow that outputs mp4/webm/gif. "
            f"First image output: {image_candidates[0]}"
        )

    raise RuntimeError(f"No downloadable video outputs found in ComfyUI history: {json.dumps(history)[:2000]}")


def download_output(server: str, output_ref: Dict[str, str], output_path: Path) -> None:
    query = urllib.parse.urlencode(output_ref)
    url = _join_url(server, f"/view?{query}")
    with urllib.request.urlopen(url, timeout=60) as response:
        output_path.write_bytes(response.read())


def _normalize_loopback_server(server: str) -> str:
    parsed = urllib.parse.urlparse(server.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"ComfyUI server must be an http(s) URL, got: {server}")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"ComfyUI server must be loopback/local, got: {server}")
    return server.rstrip("/")


def _join_url(server: str, path: str) -> str:
    return f"{server.rstrip('/')}/{path.lstrip('/')}"


def _json_request(request: urllib.request.Request) -> Dict[str, Any]:
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_get(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _replace_tokens(value: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for key, replacement in replacements.items():
            result = result.replace("{" + key + "}", replacement)
        return result
    if isinstance(value, list):
        return [_replace_tokens(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_tokens(item, replacements) for key, item in value.items()}
    return value


def _parse_set(raw: str) -> Tuple[List[str], str]:
    if "=" not in raw:
        raise ValueError(f"--set must be path=value, got: {raw!r}")
    path, value = raw.split("=", 1)
    parts = [part for part in path.strip().split(".") if part]
    if not parts:
        raise ValueError(f"--set path is empty: {raw!r}")
    return parts, value


def _apply_path(payload: Dict[str, Any], path: List[str], value: Any) -> None:
    cursor: Any = payload
    for part in path[:-1]:
        if isinstance(cursor, dict):
            cursor = cursor[part]
        elif isinstance(cursor, list):
            cursor = cursor[int(part)]
        else:
            raise ValueError(f"Cannot descend into {part!r} for path {'.'.join(path)}")
    last = path[-1]
    if isinstance(cursor, dict):
        cursor[last] = value
    elif isinstance(cursor, list):
        cursor[int(last)] = value
    else:
        raise ValueError(f"Cannot set path {'.'.join(path)}")


def _coerce_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if "." not in value:
            return int(value)
        return float(value)
    except ValueError:
        return value


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
