---
name: hf-space-deploy
description: First-time deployment to a new HuggingFace Space — handles auth, Space creation, and git LFS setup. For subsequent pushes, use /hf-space-release-check.
---

# HF Space Deploy

First-time setup to publish the local project to a new HuggingFace Space.
For ongoing updates (push + verify), use `/hf-space-release-check` instead — this skill only needs to be run once per Space.

## Prerequisites

- `huggingface-cli` is installed and logged in with a **write-permission token**
- Project has `app.py`, `requirements.txt`, `packages.txt`, and `README.md` with valid HF Space frontmatter
- Local smoke test passes (see CLAUDE.md build commands)

## Steps

### 1. Check HF auth

Run `huggingface-cli whoami`. If not logged in or missing write permission, ask the user for a write token from https://huggingface.co/settings/tokens and log in with `--add-to-git-credential`.

### 2. Confirm Space name

Ask the user for the new Space name. Derive the full repo ID as `<username>/<space-name>`.

### 3. Create the Space

Use `huggingface_hub.HfApi().create_repo()` with `repo_type="space"`, `space_sdk="gradio"`, `exist_ok=True`. Run inside the project's conda environment.

### 4. Set up git and LFS

Initialize git in the project directory, set the HF Space URL as remote `origin`. Configure git LFS to track `*.onnx` and `*.png` before any commits:
```bash
git lfs install
git lfs track "*.onnx"
git lfs track "*.png"
git add .gitattributes
```

### 5. Run /hf-space-release-check

Run the full pre-push checklist now. It will handle the commit, push, and post-push verification. The first push uploads LFS objects (glasses images + ONNX model) and may take a moment.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `libGL.so.1` error at runtime | Ensure `packages.txt` contains `libgl1` and `libglib2.0-0` |
| Space stuck in ERROR after push | Run `/debug-space` |
| Storage keeps growing | Confirm `save_frame()` in `app.py` uses `tempfile.gettempdir()` |

For mediapipe failures, Gradio streaming issues, and Python version constraints, see `CLAUDE.md`.
