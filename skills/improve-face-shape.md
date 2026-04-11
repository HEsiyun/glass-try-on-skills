---
name: improve-face-shape
description: Improve face shape detection and glasses recommendation logic — works within YuNet 5-landmark constraint, no mediapipe.
---

# Improve Face Shape Detection

Improve the accuracy of face shape classification and glasses recommendations without adding new model dependencies.

## Current approach and its limitations

`determine_face_shape(fw, fh)` uses only the YuNet bounding box:
- `ratio = fw / fh`
- `> 0.90` → Round, `< 0.75` → Oval, else → Square

This is a rough heuristic. Bounding boxes vary with head tilt and detection margin — a wide hairstyle or beard will inflate `fw`. The result is unstable frame-to-frame.

## What YuNet actually gives us

5 landmark points from `face[4:14].reshape(5, 2)`:
- `pts[0]` — left eye
- `pts[1]` — right eye
- `pts[2]` — nose tip
- `pts[3]` — left mouth corner
- `pts[4]` — right mouth corner

And the face bounding box: `fx, fy, fw, fh`.

## Improvement options

**Start with Option A.** It gives the biggest quality gain for the least risk. Only move to Option B if Option A is insufficient after testing.

### Option A — Temporal smoothing (simplest, big quality gain)
Average the last N (e.g. 5) `fw/fh` ratios instead of using the raw per-frame value.
Use a `collections.deque(maxlen=5)` as module-level state.
This eliminates jitter without changing the classification logic.

### Option B — Landmark-based proportions (more accurate)
Use the 5 landmarks to compute more stable measurements:
- **Eye distance**: `np.linalg.norm(pts[1] - pts[0])` — correlates with face width
- **Eye-to-mouth distance**: `np.linalg.norm(mouth_center - eye_center)` — correlates with face height
- **Mouth width**: `np.linalg.norm(pts[4] - pts[3])` — differentiates oval vs. round

Derive a ratio from these rather than the bounding box. This is more robust to detection margin variance.

### Option C — Improve the thresholds
The current thresholds (`0.90`, `0.75`) were set without calibration.
Test against a range of webcam frames and adjust to produce more balanced category distribution across users.

## Recommendation logic

Current:
```python
def recommend_glass_shape(face_shape):
    return "Round" if face_shape == "Oval" else "Square"
```

This returns "Square" for both Round and Square faces — not ideal.
Consider:
- Round face → Angular/Square frames (contrast the roundness)
- Oval face → Most shapes work; recommend "Any" or "Rectangular"
- Square face → Round or oval frames (soften the angles)

## Steps

1. Read the current `determine_face_shape` and `recommend_glass_shape` in `app.py`.
2. Decide which option(s) above to implement.
3. Implement the change with minimal scope — do not restructure `process_frame` unless using `/refactor-overlay-pipeline`.
4. Test locally: check that the textboxes update frame-by-frame and show stable, plausible labels.
5. Verify syntax and smoke test before pushing.
