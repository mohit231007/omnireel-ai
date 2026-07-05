# OmniReel Video Backend Modes

OmniReel now supports three backend modes through the main `run.py` CLI.

## 1. Procedural QA backend

Fast deterministic motion for local QA without model weights:

```powershell
python run.py `
  --topic "Show gravity by making an apple fall for a class 5 student in India" `
  --video-backend procedural `
  --no-llm-planning `
  --duration-seconds 3 `
  --output-dir outputs\motion_gravity_fast_3s `
  --font C:\Windows\Fonts\arial.ttf `
  --width 720 `
  --height 1280 `
  --rust-threads 4 `
  --log-level INFO
```

This is the fastest smoke test. It does not use a generative video model.

## 2. Real local ComfyUI backend

Real local model video generation through a loopback-only ComfyUI server:

```powershell
python run.py `
  --topic "Show gravity by making an apple fall for a class 5 student in India" `
  --video-backend comfyui `
  --comfyui-workflow workflows\real_video_workflow_api.json `
  --comfyui-server http://127.0.0.1:8188 `
  --no-llm-planning `
  --duration-seconds 3 `
  --fps 8 `
  --width 384 `
  --height 640 `
  --output-dir outputs\real_comfyui_gravity `
  --font C:\Windows\Fonts\arial.ttf `
  --rust-threads 4 `
  --comfyui-timeout 3600 `
  --log-level INFO
```

This mode:

1. Generates or deterministically creates the scene plan.
2. Creates a local base image.
3. Generates offline TTS.
4. Sends the prompt and base image to ComfyUI.
5. Downloads the real model video.
6. Burns subtitles and muxes audio through Rust.

The tested LTX workflow uses these default node patches:

```text
6.inputs.text={prompt}
7.inputs.text={negative_prompt}
78.inputs.image={input_image}
80.inputs.fps={fps}
81.inputs.filename_prefix={filename_prefix}
```

Additional workflow-specific patches can be supplied with repeated `--comfyui-set` arguments.

## 3. External LivePortrait/local animation backend

This remains available for user-provided local animation commands:

```powershell
python run.py `
  --topic "Explain photosynthesis" `
  --video-backend liveportrait `
  --diffusion-model C:\models\local-diffusion-model `
  --liveportrait-cmd "python C:\LivePortrait\run.py --image {image} --audio {audio} --output {output}" `
  --output-dir outputs\liveportrait_test
```

## Safety defaults

The ComfyUI mode applies a child-safe negative prompt by default. For educational/kids content, keep the default negative prompt unless a workflow requires custom wording.

## CPU expectations

The validated CPU-only LTX smoke test at reduced settings took around 12-13 minutes. Use low settings on CPU:

```text
width=384
height=640
fps=8
steps=4 in the ComfyUI workflow
```

For production speed, use a supported GPU and increase settings gradually.
