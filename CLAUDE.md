# Glass Try-On — Project Knowledge

## What this project is

A real-time virtual glasses try-on app running as a **Hugging Face Space** (Gradio webcam app).
Users open their webcam, see glasses overlaid on their face in real-time, can switch glasses styles,
apply photo filters, and get a lip color overlay.

HF Space: https://huggingface.co/spaces/SiyunHE/glass-try-on
Repo (local): ~/glass-try-on

---

## Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| UI / serving | Gradio (Blocks) | Webcam streaming via `gr.Image(streaming=True)` |
| Face detection | YuNet ONNX | `face_detection_yunet_2023mar.onnx`, loaded via `cv2.FaceDetectorYN` |
| Face landmarks | YuNet 5-point (current) | See "Known constraints" — mediapipe unavailable on HF Linux |
| Image processing | OpenCV headless + NumPy | No GUI deps |
| Deployment | HF Spaces (cpu-basic) | Free tier, Python 3.10 pinned in README.md |
| Conda env (local) | `glass-try-on` (Python 3.10) | |

---

## File structure

```
glass-try-on/
├── CLAUDE.md                          ← you are here
├── app.py                             ← single entrypoint, Gradio app
├── requirements.txt                   ← HF build deps
├── packages.txt                       ← system deps (libgl1, libglib2.0-0)
├── README.md                          ← HF Space metadata + description
├── face_detection_yunet_2023mar.onnx  ← YuNet model (git LFS)
├── glasses/                           ← RGBA PNGs, named glass1.png … glass7.png
└── skills/                            ← Claude Code skill files
```

---

## Architecture decisions

**Realtime latency is the priority.** Don't add models or processing steps that break
the webcam stream's responsiveness. Prefer simpler, faster approximations over accuracy.

**Single-file app.** `app.py` is intentionally monolithic for now. Modularisation is a
future goal tracked in `/refactor-overlay-pipeline`, not an immediate requirement.

**When NOT to refactor:**
- Do not split `app.py` into multiple modules unless explicitly requested by the user.
- Do not introduce new ML models without a clear latency justification.

**Glasses resources follow a naming convention.** Files must be `glass{n}.png` (1-indexed).
`change_glasses()` cycles through them by counting files in `glasses/`. Do not rename or
reformat without updating the loading logic.

**Frames from Gradio are RGB.** YuNet expects BGR — always convert with
`cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)` before running face detection.

**Save frames to `/tmp/`.** Writing to the project root fills up HF Space disk and causes
eviction. `save_frame()` must use `tempfile.gettempdir()`.

---

## Known constraints

### mediapipe is unavailable on HF Linux
- `mp.solutions` API: fails with `AttributeError: module 'mediapipe' has no attribute 'solutions'`
- `mediapipe.tasks` (new Tasks API): fails with C shared library load error on both Python 3.10 and 3.13
- **Current workaround:** YuNet's 5 landmarks (2 eyes, nose, 2 mouth corners) are used for
  glasses positioning, face shape estimation, and lip color approximation.
- If precise lip landmarks are ever needed, consider dlib or a small ONNX landmark model
  instead of mediapipe.

### Gradio streaming API changed between versions
- Original app used Gradio 3.x/4.x: output of stream callback went back to the same `input_img` component.
- Gradio 6.x (current): separate output component (`output_img`) is required.
- Pinning `gradio==4.44.1` breaks `huggingface_hub` compatibility in the conda env.
- **Current status:** Gradio 6 + separate output component — streaming functionality
  needs verification (see `/debug-space`).

### Python version
- Pin `python_version: "3.10"` in README.md frontmatter. mediapipe (if ever re-added)
  and some OpenCV wheels have issues on 3.11+.

---

## Build commands

```bash
# Local run
conda run -n glass-try-on python app.py

# Local smoke test (no webcam needed)
conda run -n glass-try-on python -c "
import cv2, numpy, gradio
cv2.FaceDetectorYN.create('face_detection_yunet_2023mar.onnx', '', (320,320))
print('OK')
"

# Syntax check before push
conda run -n glass-try-on python -m py_compile app.py && echo 'Syntax OK'

# Deploy
git add -A && git commit -m '<message>' && git push origin main
```

---

## Workflow reminders

- **Before touching `app.py` or `requirements.txt`:** run the smoke test first to establish a baseline.
- **Before pushing:** run `python -m py_compile app.py` and the smoke test.
- **After pushing:** check `api.get_space_runtime('SiyunHE/glass-try-on')` via huggingface_hub — do not rely on WebFetch (15-min cache).
- **When changing the Gradio UI:** do not move or rename `input_img` — it is the streaming source.
  Any output must go to a separate component or back to `input_img` itself (test both patterns).
- **When adding a new package to `requirements.txt`:** verify it installs on Linux Python 3.10
  and does not pull in mediapipe or GUI OpenCV as a transitive dep.
