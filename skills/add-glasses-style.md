---
name: add-glasses-style
description: Add a new glasses style PNG to the try-on app — covers asset requirements, naming convention, and verification.
---

# Add Glasses Style

Add a new glasses image to the virtual try-on app so it appears in the rotation.

## Asset requirements

- Format: PNG with alpha channel (RGBA, 4 channels)
- Orientation: glasses facing straight forward, horizontally centered
- Background: transparent (alpha = 0 outside the frames)
- Recommended size: 400–800px wide; aspect ratio roughly 3:1 (wide, not tall)
- The overlay pipeline resizes to fit the face bounding box, so exact pixel dimensions are flexible

## Naming and placement

Files must follow the naming convention `glass{n}.png` where `n` is the next integer after the current highest.

1. Check current count: list files in `glasses/` — they are currently named `glass1.png` through `glass7.png`.
2. Name the new file `glass8.png` (or the next available number).
3. Place it in `glasses/`.

Do not skip numbers or use other names — `change_glasses()` counts total files in the directory and cycles 1-indexed.

## Channel order

`_load_glass()` in `app.py` reads the PNG with `cv2.IMREAD_UNCHANGED` (giving BGRA) and then swaps B↔R to produce RGBA:
```python
b, g, r, a = cv2.split(img)
return cv2.merge((r, g, b, a))
```
So the PNG on disk should be a standard RGBA PNG — no manual BGR reordering needed.

## Verify the new style locally

```python
import cv2
img = cv2.imread("glasses/glass8.png", cv2.IMREAD_UNCHANGED)
assert img is not None, "File not found or unreadable"
assert img.shape[2] == 4, f"Expected 4 channels (RGBA), got {img.shape[2]}"
print(f"OK: {img.shape}")
```

Run this inside the `glass-try-on` conda env before pushing.

## Git LFS

`*.png` files are tracked by git LFS (configured on the HF remote).
When staging, confirm LFS is active: `git lfs status` should list the new PNG as a LFS pointer.

If LFS is not tracking it:
```bash
git lfs track "*.png"
git add .gitattributes
```

Then stage and commit normally.

## After adding

1. Run the app locally and click "Next Glasses" until the new style appears.
2. Confirm the overlay renders correctly on a face (no transparent artifact, correct placement).
3. Commit and push: `git add glasses/glass8.png && git commit -m 'feat: add glass8 style' && git push origin main`
