# Public Sharing and LinkedIn Pack

This document is written from Mohit's point of view for publishing OmniReel AI as a portfolio-grade AI systems project.

## GitHub About section

Use this in the repo About panel:

```text
Local-first AI video generation pipeline using Python, ComfyUI, offline TTS, and a Rust/FFmpeg compositor.
```

Suggested topics:

```text
ai-video-generation
comfyui
rust
python
ffmpeg
local-ai
offline-ai
text-to-video
portfolio-project
multimodal-ai
```

## Short project description from my POV

I built OmniReel AI as a local-first AI video generation pipeline. The project takes a topic, creates a short educational scene, generates narration offline, renders a video through a local ComfyUI workflow, and uses a Rust/FFmpeg compositor to produce the final MP4 with subtitles and audio.

This was built in the same spirit as my Text-to-SQL and Text-to-Comic projects: not as a polished consumer app, but as a serious end-to-end AI systems project that proves I can connect models, tools, orchestration, media processing, and practical engineering into one working pipeline.

## Challenges I faced and how I solved them

### 1. Keeping the project local-first

The first challenge was avoiding the easy route of calling cloud APIs. I wanted the project to run locally, so I had to coordinate local Python orchestration, local ComfyUI, offline TTS, FFmpeg, and Rust instead of depending on hosted generation APIs.

I solved this by designing a local-only pipeline where ComfyUI and Ollama run on loopback addresses like `127.0.0.1`, while generated assets stay inside local output folders.

### 2. Making Python and Rust work together

Python is good for model orchestration, but video compositing and subtitle burning are better handled with a faster systems layer. I split responsibilities clearly: Python handles planning, TTS, and model/backend orchestration; Rust handles the final video/audio/subtitle composition.

This made the project more realistic than a single notebook demo.

### 3. Debugging Windows local setup

The setup involved Python virtual environments, Rust/Cargo, FFmpeg, ComfyUI, CPU-only Torch, model files, and workflow exports. A lot of the work was not just writing code, but making all the tools cooperate on a Windows machine.

I solved this by building small validation steps: first Rust smoke tests, then Python import tests, then procedural video tests, then ComfyUI manual tests, and finally full end-to-end runs.

### 4. ComfyUI workflow automation

ComfyUI worked manually, but automation required exporting the workflow in API format and discovering the correct node IDs for prompt, negative prompt, input image, FPS, and output filename.

I solved this by adding a workflow inspector that identifies the nodes and then wiring those nodes into an adapter that can patch the workflow before submitting it to ComfyUI.

### 5. Handling unsafe/default model drift

One early ComfyUI output followed an unwanted default/template visual instead of the educational prompt. That was an important lesson: model output is not safe or controlled just because the pipeline runs locally.

I fixed the immediate issue by adding prompt patching, child-safe positive prompt wrapping, and default negative prompts to reduce adult/suggestive drift in educational demos.

### 6. CPU-only performance

The real video backend works on CPU, but it is slow. Instead of hiding that, I added a procedural backend for quick QA and kept the real ComfyUI backend for heavier model-based demos.

This gave the project two useful modes: fast reproducible testing and real local model rendering.

### 7. File and artifact hygiene

Generated videos, audio, images, model weights, virtual environments, and Rust build artifacts should not be committed to git.

I handled this through `.gitignore`, local output folders, and docs that explain what should and should not be committed.

## Main LinkedIn launch post

```text
I built OmniReel AI — a local-first AI video generation pipeline.

The idea was simple: can I take a topic and generate a short narrated educational video locally, without depending on hosted AI APIs?

The final pipeline now does this:

Topic
→ local scene planning
→ local base image
→ offline TTS narration
→ ComfyUI image-to-video generation
→ subtitle timing
→ Rust + FFmpeg final video compositor
→ final MP4

This was not just a model demo. It became a proper AI systems engineering project.

Some of the hardest parts were:

1. Making the entire pipeline local-first instead of API-dependent.
2. Connecting Python orchestration with a Rust video compositor.
3. Getting ComfyUI to work from code, not just manually in the browser.
4. Exporting and patching the right ComfyUI API workflow nodes.
5. Handling unsafe/default model drift with prompt control and negative prompts.
6. Managing CPU-only performance by adding both a fast procedural backend and a real ComfyUI backend.
7. Keeping generated media, model files, and local artifacts out of git.

The project now supports multiple video backends:

• procedural backend for fast local QA
• ComfyUI backend for real local model video generation
• external animation command hook for future backends

Tech stack:
Python, Rust, FFmpeg, ComfyUI, offline TTS, local workflow automation.

This joins my Text-to-SQL and Text-to-Comic projects as part of my practical AI systems portfolio.

GitHub: <add repo link>

#AI #GenerativeAI #Python #Rust #ComfyUI #LocalAI #OpenSource #AIVideo #MachineLearning #PortfolioProject
```

## Short LinkedIn version

```text
I built OmniReel AI, a local-first AI video generation pipeline.

It converts a topic into a narrated video using:

• Python orchestration
• ComfyUI for local image-to-video generation
• Offline TTS
• Rust + FFmpeg for subtitle/audio composition
• Multi-backend CLI for procedural QA and real model rendering

The biggest challenge was making this work end-to-end locally instead of relying on cloud APIs. I had to debug ComfyUI workflow exports, patch API nodes, handle unsafe default model drift, manage CPU-only performance, and connect Python with a Rust media compositor.

This is now part of my AI systems portfolio alongside Text-to-SQL and Text-to-Comic.

GitHub: <add repo link>
```

## Portfolio case-study structure

### Title

```text
OmniReel AI — Local AI Video Generation Pipeline
```

### Problem

Most AI media demos rely on hosted APIs or isolated notebooks. I wanted to build a local-first pipeline that proves end-to-end AI systems engineering: orchestration, local model workflows, offline narration, video composition, and reproducible CLI execution.

### Solution

I built a Python + Rust pipeline that converts a topic into a narrated video. Python manages planning, local TTS, ComfyUI workflow automation, and manifests. Rust handles the final FFmpeg-based video compositor with subtitles and audio.

### Key features

- Local-first generation path.
- Real ComfyUI image-to-video backend.
- Procedural backend for fast QA.
- Offline TTS narration.
- Rust/FFmpeg video compositor.
- Subtitle burn-in.
- Reproducible CLI commands.
- Workflow inspector for ComfyUI API node mapping.

### Challenges

- ComfyUI manual workflow had to be converted into API workflow automation.
- Prompt control was required to prevent default/template drift.
- CPU-only model generation was slow, so I added a fast procedural backend for QA.
- PowerShell/Windows encoding issues required robust JSON handling.
- The system needed clean boundaries between generated media, model files, and version-controlled source code.

### Outcome

The project now has a tagged v0.1 release and can generate local narrated educational videos using a real ComfyUI backend and Rust-based final composition.

## Demo video script

1. Show the GitHub README and explain the project in one sentence.
2. Show that ComfyUI is running locally on `127.0.0.1`.
3. Show the `python run.py --video-backend comfyui` command.
4. Show the generated `final_omnireel.mp4`.
5. Briefly show the repo structure: `scripts/`, `src/`, `workflows/`, `docs/`.
6. End with the v0.1 release tag.

## What not to claim publicly

Do not claim:

- production child-safety certification;
- app-store readiness;
- real-time generation on CPU;
- model weights included in the repo;
- guaranteed safe output without human review.

Use this framing instead:

```text
A portfolio-grade local AI video pipeline and technical systems demo.
```
