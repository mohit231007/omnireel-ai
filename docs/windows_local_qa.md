# OmniReel AI Windows Local QA

This guide is optimized for Windows PowerShell.

## 1. Required local tools

Verify the tools:

```powershell
git --version
python --version
cargo --version
rustc --version
ffmpeg -version
ffprobe -version
ollama --version
```

If `cargo` or `rustc` is not recognized, install Rust with rustup from the official Rust installer. Restart PowerShell after installation so `%USERPROFILE%\.cargo\bin` is loaded into `PATH`.

## 2. Python QA

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m py_compile run.py scripts\__init__.py scripts\models_pipeline.py scripts\orchestrator.py scripts\cli.py
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 3. Rust QA

```powershell
cargo fmt --check
cargo check
cargo test
cargo build --release
.\target\release\omnireel-ai-engine.exe --help
```

If formatting fails only because files need formatting:

```powershell
cargo fmt
cargo fmt --check
```

## 4. Rust compositor smoke test

```powershell
mkdir outputs\smoke -Force

ffmpeg -y -f lavfi -i "color=c=0x202020:s=720x1280:d=5:r=30" `
  -c:v libx264 -pix_fmt yuv420p outputs\smoke\raw_animated.mp4

ffmpeg -y -f lavfi -i "sine=frequency=440:duration=5" `
  -c:a pcm_s16le outputs\smoke\dialogue.wav

@'
{
  "segments": [
    {
      "start": 0.2,
      "end": 2.3,
      "text": "OmniReel AI local Rust subtitle test"
    },
    {
      "start": 2.5,
      "end": 4.7,
      "text": "If you can read this, the compositor works"
    }
  ]
}
'@ | Set-Content -Encoding UTF8 outputs\smoke\whisper_timestamps.json

.\target\release\omnireel-ai-engine.exe `
  --input-video outputs\smoke\raw_animated.mp4 `
  --audio outputs\smoke\dialogue.wav `
  --subtitles outputs\smoke\whisper_timestamps.json `
  --output outputs\smoke\final_omnireel.mp4 `
  --font C:\Windows\Fonts\arial.ttf `
  --fps 30 `
  --threads 4

Start-Process outputs\smoke\final_omnireel.mp4
```

## 5. Rust-only QA through Python runner

```powershell
@'
{
  "output_dir": "outputs/smoke",
  "plan_json": "outputs/smoke/scene_plan.json",
  "base_image": "outputs/smoke/base_image.png",
  "audio": "outputs/smoke/dialogue.wav",
  "raw_video": "outputs/smoke/raw_animated.mp4",
  "subtitles_json": "outputs/smoke/whisper_timestamps.json",
  "final_video": "outputs/smoke/final_from_runpy.mp4",
  "font_path": "C:/Windows/Fonts/arial.ttf",
  "fps": 30
}
'@ | Set-Content -Encoding UTF8 outputs\smoke\manifest.json

python run.py `
  --topic "smoke test" `
  --skip-python-generation `
  --existing-manifest outputs\smoke\manifest.json `
  --rust-threads 4 `
  --log-level INFO

Start-Process outputs\smoke\final_from_runpy.mp4
```
