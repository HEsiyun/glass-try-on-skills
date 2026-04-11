---
name: glass-tryon-setup
description: Set up the Glasses Virtual Try-On project from scratch — creates conda env, downloads assets, and generates all project files from spec.
---

# Glass Try-On: Project Setup

Generate all project files and set up the environment for the glasses virtual try-on app.
See `CLAUDE.md` for stack details, known constraints, and build commands.

## Step 1 — Conda environment

Create a Python 3.10 conda environment named `glass-try-on`.
(Python 3.10 is required — see CLAUDE.md "Python version" constraint.)

## Step 2 — Download assets

If `glasses/` or `face_detection_yunet_2023mar.onnx` are missing, download from `SiyunHE/glass_try_on1` (HF Space):
- `glasses/glass1.png` through `glasses/glass7.png`
- `face_detection_yunet_2023mar.onnx`

Use `huggingface-cli download --repo-type space --local-dir <project_dir>`, then remove the `.cache/` folder it creates.

## Step 3 — Generate `requirements.txt`

```
gradio
opencv-python-headless
numpy
```

Do NOT add mediapipe — it is broken on HF Linux. See CLAUDE.md for details.

## Step 4 — Generate `packages.txt`

System dependencies for the HF Linux environment:
```
libgl1
libglib2.0-0
```

## Step 5 — Generate `README.md`

HF Space metadata header followed by a brief description:
- title: Glasses Virtual Try-On
- emoji: 🕶️
- colorFrom: purple, colorTo: blue
- sdk: gradio
- app_file: app.py
- python_version: "3.10"

## Step 6 — Generate `app.py`

Write a Gradio webcam app with the following specification.

### Libraries
`cv2`, `numpy`, `os`, `tempfile`, `gradio`, `datetime`
No mediapipe.

### Face detection
Load `face_detection_yunet_2023mar.onnx` via `cv2.FaceDetectorYN`. Set input size per frame dynamically.
Frames from Gradio are RGB — convert to BGR before calling `face_detector.detect()`.

### Glasses overlay
- Load glasses as RGBA PNGs from `glasses/glass{n}.png`, swapping B↔R channels on load (OpenCV reads as BGRA)
- Global `num` (1-indexed) and `overlay` track current glass
- `change_glasses()` increments `num` cyclically through all files in `glasses/`
- `overlay_png(background, fg, pos)`: custom alpha-blend — clips to frame bounds, blends using `fg`'s alpha channel
- In `process_frame()`: resize overlay to face bounding box, rotate to match eye angle, place centered between the eyes

### Face shape detection
Using YuNet bounding box: `fw/fh > 0.90` → Round, `< 0.75` → Oval, else → Square.
For improvements, see `/improve-face-shape`.

### Lip color
10 color options using YuNet mouth corner landmarks (pts[3], pts[4]) to approximate the lip region via ellipses.

### Photo filters
cartoon, edges, sepia, negative, sketch, blur — applied only when lip color is "none"

### Save frame
Write to `tempfile.gettempdir()` — never to the project root.

### Gradio UI
- `gr.Blocks` with Soft purple/blue theme
- `input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True)`
- Two dropdowns: filter and lip color (mutually exclusive)
- "Next Glasses" button
- Two textboxes: detected face shape, recommended glass shape
- "Save as Picture" button with file download
- `input_img.stream(webcam_input, [input_img, transform, lip_color], [input_img, face_shape_out, glass_shape_out], stream_every=0.1)`

## Step 7 — Install dependencies

```bash
conda run -n glass-try-on pip install gradio opencv-python-headless numpy
```

## Step 8 — Verify locally before deploying

See CLAUDE.md build commands for the smoke test and syntax check.
Run `app.py` locally and confirm `http://localhost:7860` works with webcam streaming.

Only proceed to `/hf-space-deploy` after both checks pass.

## Done

Report that the project is ready and all files have been generated. Proceed with `/hf-space-deploy` to publish.
