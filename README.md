# OmniReel AI

A 100% local AI video-generation pipeline for educational explainers and portfolio-grade AI engineering demos.

OmniReel turns a short topic into a narrated video using local planning, local image/video generation, offline text-to-speech, and a Rust compositor for subtitles/audio. It is designed in the same spirit as practical AI portfolio projects such as Text-to-SQL and Text-to-Comic: a working end-to-end system that demonstrates architecture, orchestration, model integration, and production-minded engineering.

## What it does

```text
topic
  -> local lesson/scene plan
  -> local base image
  -> offline narration
  -> local video backend
  -> subtitle timing JSON
  -> Rust subtitle burn + audio mux
  -> final MP4
```

## Current status

| Area | Status |
| --- | --- |
| Python orchestration | Working |
| Rust video compositor | Working |
| Offline TTS | Working with pyttsx3; Piper hook available |
| Procedural QA backend | Working |
| ComfyUI real video backend | Working through exported API workflow |
| Main CLI backend switch | Working via `--video-backend` |
| App-store/mobile product | Not the target of this repo yet |

## Backend modes

OmniReel supports three video backend modes through `run.py`:

```text
--video-backend procedural   # fast deterministic local QA/demo
--video-backend comfyui      # real local ComfyUI video model backend
--video-backend liveportrait # external local animation command hook
```

## Fast local QA demo

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

## Real local ComfyUI video demo

Start ComfyUI locally first:

```powershell
cd C:\ComfyUI
.\.venv\Scripts\Activate.ps1
python main.py --listen 127.0.0.1 --port 8188 --disable-api-nodes --cpu
```

Then run OmniReel:

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

Open the final video:

```powershell
Start-Process outputs\real_comfyui_gravity\final_omnireel.mp4
```

## Local-first design

OmniReel is intentionally local-first:

- no hosted LLM API required;
- no cloud video API required;
- no remote generation endpoint required;
- local ComfyUI server is restricted to loopback URLs;
- generated media stays in `outputs/`, which is ignored by git.

## Repository map

```text
scripts/cli.py                     Main run.py CLI
scripts/models_pipeline.py         Local planning, base image, TTS, subtitles, procedural backend
scripts/comfyui_video_client.py    Loopback-only ComfyUI API adapter
scripts/run_comfyui_omnireel.py    End-to-end ComfyUI pipeline runner
scripts/comfyui_workflow_inspector.py  API workflow inspector
src/video.rs                       Rust video decode/encode/audio mux
src/subtitles.rs                   Subtitle parsing and burn-in
src/main.rs                        Rust compositor CLI
docs/                              Setup, backend, and QA notes
```

## Portfolio positioning

This is a portfolio-grade AI systems project, not a polished consumer app. It demonstrates:

- local LLM orchestration;
- local image/video model integration;
- ComfyUI workflow automation;
- Python-to-Rust pipeline boundaries;
- FFmpeg-based media engineering;
- offline TTS and subtitle timing;
- safety-aware prompt wrapping for educational content;
- reproducible command-line demos.

## Notes

- Model weights are not committed to this repo.
- Generated videos/audio/images are not committed to this repo.
- CPU-only ComfyUI video generation is slow; use reduced settings for demos.
- The current educational demo is a technical showcase, not a certified child-safety product.
