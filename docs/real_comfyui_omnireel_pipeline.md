# Real ComfyUI OmniReel Pipeline

This path runs the complete local model-backed pipeline:

1. Local scene planning or deterministic planning.
2. Local procedural base image for image-to-video conditioning.
3. Offline TTS narration.
4. Real local ComfyUI LTX image-to-video render.
5. Local subtitle timings.
6. Rust subtitle burn and audio mux.

This is different from `--qa-static-assets`: the final motion video comes from the real ComfyUI video model workflow.

## Requirements

ComfyUI must be running locally:

```powershell
cd C:\ComfyUI
.\.venv\Scripts\Activate.ps1
python main.py --listen 127.0.0.1 --port 8188 --disable-api-nodes --cpu
```

The workflow must be exported in API format:

```text
workflows\real_video_workflow_api.json
```

For the tested LTX workflow, the expected API node patches are:

```text
6.inputs.text={prompt}
7.inputs.text={negative_prompt}
78.inputs.image={input_image}
80.inputs.fps={fps}
81.inputs.filename_prefix={filename_prefix}
```

## Run the real model pipeline

From the OmniReel repo:

```powershell
cd C:\Users\mohit\OmniReelQA\omnireel-ai
.\.venv\Scripts\Activate.ps1

git pull origin main

python -m py_compile `
  scripts\run_comfyui_omnireel.py `
  scripts\comfyui_video_client.py `
  scripts\models_pipeline.py `
  scripts\orchestrator.py

python -m scripts.run_comfyui_omnireel `
  --topic "Show gravity by making an apple fall for a class 5 student in India" `
  --workflow workflows\real_video_workflow_api.json `
  --server http://127.0.0.1:8188 `
  --no-llm-planning `
  --duration-seconds 3 `
  --fps 8 `
  --width 384 `
  --height 640 `
  --output-dir outputs\real_comfyui_gravity `
  --font C:\Windows\Fonts\arial.ttf `
  --rust-threads 4 `
  --timeout 3600 `
  --log-level INFO

Start-Process outputs\real_comfyui_gravity\final_omnireel.mp4
```

## CPU warning

This is expected to be slow on CPU-only hardware. The first validated LTX smoke test took about 12-13 minutes for 4 sampling steps at reduced settings.
