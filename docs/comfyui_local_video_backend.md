# ComfyUI Local Video Backend

This is the production-direction backend for real local model video.

`--qa-static-assets` is useful for diagnostics and fast demos, but it is deterministic procedural animation. For real model output, run a local ComfyUI server with a video workflow and call it through `scripts/comfyui_video_client.py`.

## What this adapter does

The adapter:

1. Connects only to a loopback ComfyUI server such as `http://127.0.0.1:8188`.
2. Loads an API-format workflow JSON.
3. Replaces placeholders such as `{prompt}`, `{input_image}`, `{duration_seconds}`, `{fps}`, `{width}`, and `{height}`.
4. Optionally uploads an input image to ComfyUI.
5. Queues the workflow through ComfyUI's local API.
6. Waits for completion.
7. Downloads the first video output to the requested MP4/WebM/GIF path.

## Workflow requirements

Use an API-format ComfyUI workflow that outputs a video file such as MP4, WebM, or GIF.

The workflow may include placeholders in string fields:

```text
{prompt}
{negative_prompt}
{input_image}
{filename_prefix}
{seed}
{duration_seconds}
{fps}
{width}
{height}
```

For node-specific patching, pass repeated `--set` values:

```powershell
--set "6.inputs.text={prompt}" `
--set "7.inputs.text={negative_prompt}" `
--set "12.inputs.image={input_image}" `
--set "30.inputs.filename_prefix={filename_prefix}"
```

## Start ComfyUI locally

Run ComfyUI locally and keep the server on loopback:

```powershell
cd C:\ComfyUI
python main.py --listen 127.0.0.1 --port 8188 --disable-api-nodes
```

## Test adapter directly

```powershell
python scripts\comfyui_video_client.py `
  --server http://127.0.0.1:8188 `
  --workflow workflows\your_video_workflow_api.json `
  --input-image outputs\motion_gravity_fast_3s\base_image.png `
  --prompt "A red apple falling down in a classroom science animation" `
  --negative-prompt "blurry, low quality, distorted" `
  --duration-seconds 3 `
  --fps 30 `
  --width 720 `
  --height 1280 `
  --output outputs\comfyui_real_motion\raw_model_video.mp4
```

## Use through OmniReel today

Until the first built-in ComfyUI mode is fully wired into `run.py`, use the existing external animation command boundary. The command receives `{image}`, `{audio}`, and `{output}` from OmniReel. For more control, run the adapter directly and then use `--skip-python-generation` with a manifest.

## Security rule

Keep ComfyUI on `127.0.0.1` and use `--disable-api-nodes` unless a workflow explicitly needs online API nodes. OmniReel's adapter rejects non-loopback ComfyUI server URLs.
