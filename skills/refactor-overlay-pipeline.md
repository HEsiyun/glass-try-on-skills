---
name: refactor-overlay-pipeline
description: Refactor the glasses overlay and face processing pipeline in app.py — safely decouple detection, landmark extraction, and rendering.
---

# Refactor Overlay Pipeline

Refactor the main processing pipeline in `app.py` without breaking streaming latency or the Gradio wiring.

## When to use this skill

- The pipeline is getting hard to follow (detection, landmark math, overlay, filter, lip color all tangled in one function)
- Adding a new overlay type (e.g., hat, earrings) requires touching unrelated logic
- You want to swap the face detector without rewriting everything

## Constraints

- **Latency is the priority.** Do not add model calls, disk reads, or heavy compute per frame.
- `app.py` stays as the single entrypoint — do not split into multiple modules unless the file exceeds ~400 lines and the user agrees.
- The Gradio `.stream()` wiring must not change:
  - Input: `[input_img, transform, lip_color]`
  - Output: `[input_img, face_shape_out, glass_shape_out]`
  - `webcam_input(frame, transform, lip_color)` remains the top-level callback

## Proposed structure (reference only — adapt to what's actually in app.py)

```
webcam_input(frame, transform, lip_color)
  └─ process_frame(frame)          → (frame_rgb, detections)
       ├─ detect_faces(frame_bgr)  → list of face dicts {box, landmarks, angle}
       └─ for each face:
            ├─ place_overlay(frame_rgb, overlay, face)   → frame_rgb
            └─ extract_face_meta(face)                   → (face_shape, glass_shape)
  └─ (if transform) apply_filter(frame_rgb, transform)  → frame_rgb
  └─ (if lip_color) change_lip_color(frame_rgb, lip_color, mouth_pts) → frame_rgb
```

## Steps

Refactor in small increments. After each step, verify behavior is unchanged before proceeding.

1. **Read `app.py` fully** before touching anything.
2. Extract `detect_faces(frame_bgr)` — isolates YuNet logic. **Verify:** face detection still works locally.
3. Extract `place_overlay(frame_rgb, overlay_rgba, face_dict)` — isolates rotate+blend. **Verify:** glasses overlay still renders correctly.
4. Keep `apply_filter` and `change_lip_color` as-is (already isolated).
5. Update `process_frame()` to call the new helpers. **Verify:** all features still work end-to-end.
6. Do not change function signatures visible to the Gradio layer (`webcam_input`, `change_glasses`, `save_frame`).

## Verification after refactor

```bash
conda run -n glass-try-on python -m py_compile app.py && echo 'Syntax OK'
```

Then run the full smoke test (see CLAUDE.md build commands) and test locally at `http://localhost:7860`:
- Glasses overlay renders correctly
- "Next Glasses" button works
- Filter dropdown applies a filter
- Lip color dropdown applies lip color
- Face shape textboxes update

Only push after all of the above pass locally.
