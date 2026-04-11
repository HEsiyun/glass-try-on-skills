---
name: hf-space-release-check
description: Pre-release checklist before pushing changes to the HF Space — syntax, imports, local run, streaming, and git hygiene.
---

# HF Space Release Check

Run this checklist before every push to `origin main`.
**If any step fails → STOP. Fix the issue before continuing. Do not push until all steps pass.**

## 1. Syntax check

```bash
conda run -n glass-try-on python -m py_compile app.py && echo 'Syntax OK'
```

If this fails, fix the syntax error before continuing.

## 2. Import and model smoke test

```bash
conda run -n glass-try-on python -c "
import cv2, numpy, gradio
cv2.FaceDetectorYN.create('face_detection_yunet_2023mar.onnx', '', (320,320))
print('OK')
"
```

If this fails:
- `ModuleNotFoundError` → package missing from conda env or requirements.txt
- `.onnx` load error → model file missing or corrupt (re-download from HF LFS)
- `libGL` error → only happens on Linux; packages.txt should cover it

## 3. Local run check

```bash
conda run -n glass-try-on python app.py
```

Open `http://localhost:7860`. Verify:
- [ ] Page loads without JS errors
- [ ] Webcam stream starts
- [ ] Glasses appear on face
- [ ] "Next Glasses" cycles to a different style
- [ ] Filter dropdown applies a visible effect
- [ ] Lip color dropdown applies a visible lip tint
- [ ] "Save as Picture" downloads a file
- [ ] Face shape and recommended shape textboxes update

## 4. Requirements hygiene

- No package in `requirements.txt` pulls in mediapipe, GUI OpenCV, or any package with a known HF Linux build failure.
- Versions are either unpinned (get latest) or explicitly pinned and verified on Linux Python 3.10.
- `gradio` is compatible with `huggingface_hub` in the conda env (test with `import gradio` — no ImportError).

## 5. Git hygiene

- `git status` shows only the files you intended to change.
- `*.onnx` and `*.png` files appear as LFS pointers in `git lfs status`, not as raw binary blobs.
- Commit message follows the pattern: `fix: ...` / `feat: ...` / `refactor: ...`

## 6. Push and verify

```bash
git push origin main
```

After pushing:
- Wait ~60s for HF to build.
- Check runtime via `huggingface_hub.HfApi().get_space_runtime('SiyunHE/glass-try-on')`.
- Status must reach `RUNNING`. If it stays `BUILDING` or goes to `ERROR`, run `/debug-space`.
