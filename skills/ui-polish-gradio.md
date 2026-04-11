---
name: ui-polish-gradio
description: Improve the Gradio UI — layout, labels, instructions — without breaking streaming wiring or webcam functionality.
---

# UI Polish (Gradio)

Make visual and UX improvements to the Gradio interface without breaking the streaming pipeline.

## Hard constraints — do not change these

- `input_img` component identity must remain: it is the webcam source AND the stream output target.
  Do not rename, remove, or change its `sources` or `type` parameters.
- The `.stream()` call signature must remain:
  ```python
  input_img.stream(webcam_input,
                   [input_img, transform, lip_color],
                   [input_img, face_shape_out, glass_shape_out],
                   stream_every=0.1)
  ```
- Do not add new output components to the stream outputs list without also updating `webcam_input()` return value.
- Do not upgrade or downgrade the `gradio` version — changes to streaming behavior are version-sensitive.

## Safe things to change

- Markdown text: headings, descriptions, instructions
- Component labels and placeholder text
- Layout: `gr.Row`, `gr.Column`, `gr.Tab`, `gr.Accordion` wrappers — as long as all components are still present
- Theme parameters: `primary_hue`, `secondary_hue`, `font`
- Button labels
- Dropdown choice display names (but do NOT change the underlying values — `webcam_input` uses them as keys)
- Add static images (e.g., example glasses previews) using `gr.Image(value=path)` — these are not part of the stream

## Adding a new interactive control

If adding a new dropdown or checkbox that feeds into `webcam_input`:
1. Add the component.
2. Add it to the stream inputs list: `[input_img, transform, lip_color, new_control]`
3. Add the parameter to `webcam_input(frame, transform, lip_color, new_control)`.
4. Test locally — changing the stream inputs list is the most common source of breakage.

## After any UI change

Test the streaming path locally:
1. `conda run -n glass-try-on python app.py`
2. Open `http://localhost:7860`, start webcam.
3. Confirm the video stream is live and glasses still overlay correctly.
4. Confirm all buttons and dropdowns respond.

Then run `/hf-space-release-check` before pushing.
