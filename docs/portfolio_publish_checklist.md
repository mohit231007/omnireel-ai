# Portfolio Publish Checklist

This checklist is for publishing OmniReel as a GitHub/portfolio project, similar to a Text-to-SQL or Text-to-Comic project. It is not an app-store production checklist.

## Publish target

Recommended label:

```text
OmniReel AI v0.1 - Local AI Video Pipeline Demo
```

## Must-have before sharing the repo

- [x] Working local procedural QA demo.
- [x] Working Rust compositor.
- [x] Working offline TTS path.
- [x] Working ComfyUI adapter.
- [x] Working main CLI `--video-backend comfyui` path.
- [x] README explains portfolio positioning.
- [ ] Commit `workflows/real_video_workflow_api.json` if the exported workflow is clean and contains no private/local-only image names that should be removed.
- [ ] Add one short demo video or GIF to the portfolio page, not necessarily to git.
- [ ] Add 3-5 screenshots to the portfolio page, not necessarily to git.
- [ ] Add a short architecture diagram.
- [ ] Add a 60-90 second screen-recorded walkthrough.

## Demo script

1. Show the CLI command.
2. Show ComfyUI running locally on `127.0.0.1`.
3. Run the procedural backend for fast proof.
4. Show the real ComfyUI backend command.
5. Show the final MP4 with narration and subtitles.
6. Explain Python vs Rust responsibilities.

## Resume bullets

- Built a 100% local AI video-generation pipeline using Python orchestration, ComfyUI model automation, offline TTS, and a Rust/FFmpeg compositor.
- Integrated a real local image-to-video workflow with prompt patching, loopback-only API validation, subtitle generation, and final MP4 rendering.
- Designed a multi-backend CLI supporting procedural QA, external animation hooks, and real ComfyUI video generation for reproducible AI media demos.

## LinkedIn/GitHub summary

OmniReel AI is a local-first AI video pipeline that converts educational topics into narrated videos using Python, ComfyUI, offline TTS, and a Rust media compositor. The project demonstrates practical AI systems engineering: model orchestration, workflow automation, safety-aware prompting, local inference constraints, and multimedia processing.

## Not required for portfolio publish

- Play Store release.
- iOS release.
- Parent dashboard.
- Kiosk/screen-lock mode.
- Certified child-safety pipeline.
- Production-scale GPU deployment.

Those belong to a future product roadmap, not the v0.1 portfolio demo.
