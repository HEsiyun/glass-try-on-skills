---
name: debug-space
description: Diagnose and fix a broken HF Space — covers build failures, streaming issues, and feature regressions.
---

# Debug Space

Systematically diagnose and fix the HF Space at `SiyunHE/glass-try-on`.

## Step 1 — Get current Space status

Use `huggingface_hub.HfApi().get_space_runtime('SiyunHE/glass-try-on')` to fetch live status.
Do NOT use WebFetch — it has a 15-min cache and will show stale data.

If status is not `RUNNING`, go to **Build failure** below.
If status is `RUNNING` but features are broken, go to **Runtime feature failure** below.

---

## Build failure

1. Fetch the build log from the Space to find the failing line.
2. Common causes and fixes:
   - Missing system dep → add to `packages.txt`
   - Package install error → check `requirements.txt` for version conflicts or Linux 3.10 incompatibility
   - Import error at startup → run the smoke test locally first (see CLAUDE.md build commands)
   - `libGL.so.1` error → ensure `packages.txt` contains `libgl1` and `libglib2.0-0`
3. After fixing, run `python -m py_compile app.py` and the smoke test locally, then push.

---

## Runtime feature failure

Features (glasses overlay, filters, lip color) not working despite RUNNING status.

**Before touching any code:** rank the hypotheses below by likelihood given what you already know (error messages, recent changes, git diff). Start with the most probable cause and test one fix at a time. Do not apply multiple fixes blindly — it makes it impossible to know what actually worked.

### 1. Check Gradio streaming wiring

In `app.py`, find the `.stream()` call. Verify:
- **Input** is `input_img` (the webcam component)
- **Outputs** list matches exactly what `webcam_input()` returns — currently `[input_img, face_shape_out, glass_shape_out]`
- `webcam_input()` returns a 3-tuple: `(frame, face_shape, glass_shape)`
- No `output_img` component is declared but not connected — that causes silent drops

In Gradio 6, the stream output can go back to the same `input_img` component. Confirm this is wired correctly and not accidentally split.

### 2. Check frame color space

`process_frame()` receives RGB from Gradio. YuNet expects BGR.
Verify the line `frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)` is present before `face_detector.detect()`.
Also verify `overlay_png()` is called on the RGB frame (not frame_bgr).

### 3. Check glasses overlay

- Run `_load_glass(1)` locally and confirm the returned array has shape `(H, W, 4)` (RGBA).
- Confirm `overlay_png()` blends using `crop[:, :, 3:4] / 255.0` as alpha — integer division would silently break this.
- Confirm `overlay` global is being mutated by `change_glasses()` correctly.

### 4. Check lip color

`change_lip_color()` needs `mouth_pts` which comes from `pts[3]` and `pts[4]` in YuNet's 5-point output.
Verify `faces is not None` and that a face is actually being detected (add a temporary print if needed).

### 5. Local reproduction

Run the app locally with:
```
conda run -n glass-try-on python app.py
```
Open `http://localhost:7860` in a browser, enable webcam, and observe:
- Does the video stream appear in the output?
- Are glasses drawn when a face is detected?
- Does "Next Glasses" change the overlay?

Fix locally first. Only push after local tests pass.

---

## After fixing

1. `conda run -n glass-try-on python -m py_compile app.py && echo OK`
2. Run smoke test (see CLAUDE.md)
3. `git add app.py requirements.txt && git commit -m 'fix: ...' && git push origin main`
4. Re-check Space runtime status after ~60s.
