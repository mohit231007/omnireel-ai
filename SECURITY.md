# Security Policy

OmniReel AI is a local-first technical portfolio project. It is not a hosted SaaS service and does not require cloud AI APIs for the demonstrated pipeline.

## Security scope

The current security scope covers:

- local Python orchestration;
- local Rust/FFmpeg media composition;
- local ComfyUI integration over loopback URLs only;
- generated artifacts written to local `outputs/` paths;
- prevention of accidental model/media/secrets commits.

## Local-only network policy

OmniReel should only connect to services running on the local machine for generation flows.

Allowed examples:

```text
http://127.0.0.1:8188
http://localhost:8188
http://127.0.0.1:11434
```

Disallowed examples for generation backends:

```text
https://api.example.com
http://192.168.x.x:8188
http://public-host:8188
```

## Secrets policy

Do not commit:

- `.env` files;
- API keys;
- private model tokens;
- personal file paths with sensitive names;
- generated videos, audio, or screenshots containing private data;
- model weights or checkpoints.

The repository `.gitignore` excludes common local artifacts including `.env`, `.venv/`, `target/`, `outputs/`, and generated media.

## Model and workflow policy

Model weights are intentionally not included in this repository. Users must download or install their own local models according to the model owners' licenses.

ComfyUI workflow JSON files may be committed only when they are lightweight API workflow definitions and do not contain private images, secret paths, or personal data.

## Educational content safety

The current project uses safety-aware prompt wrapping and negative prompts for the real video backend, but it is not a certified child-safety system.

For public portfolio demos:

- use non-personal educational prompts;
- prefer object-only scenes such as apples, planets, balls, diagrams, and classroom boards;
- avoid prompts involving real children, identifiable people, politics, medical claims, violence, or adult themes;
- manually review all generated output before publishing.

## Reporting issues

Open a GitHub issue with:

- the command used;
- OS and Python/Rust versions;
- whether the backend was `procedural`, `comfyui`, or `liveportrait`;
- relevant logs with secrets and private paths removed.

Do not paste API keys, private model tokens, or private personal data into public issues.
