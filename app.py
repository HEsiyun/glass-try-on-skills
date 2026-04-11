import cv2
import numpy as np
import os
import tempfile
import gradio as gr
from datetime import datetime

# ── Models ────────────────────────────────────────────────────────────────────
face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))

# ── Glasses state ─────────────────────────────────────────────────────────────
num = 1

def _load_glass(n):
    img = cv2.imread(f"glasses/glass{n}.png", cv2.IMREAD_UNCHANGED)
    b, g, r, a = cv2.split(img)
    return cv2.merge((r, g, b, a))

overlay = _load_glass(num)
total_glass_num = sum(len(f) for _, _, f in os.walk("glasses"))


# ── Helpers ───────────────────────────────────────────────────────────────────
def overlay_png(background, fg, pos):
    x, y = pos
    h, w = fg.shape[:2]
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    oy1, ox1 = max(0, -y), max(0, -x)
    oy2, ox2 = oy1 + (y2 - y1), ox1 + (x2 - x1)
    if y2 <= y1 or x2 <= x1:
        return background
    crop = fg[oy1:oy2, ox1:ox2]
    alpha = crop[:, :, 3:4] / 255.0
    background[y1:y2, x1:x2] = (
        alpha * crop[:, :, :3] + (1 - alpha) * background[y1:y2, x1:x2]
    ).astype(np.uint8)
    return background


def change_glasses():
    global num, overlay
    num = num % total_glass_num + 1
    overlay = _load_glass(num)


def determine_face_shape(fw, fh):
    ratio = fw / fh
    if ratio > 0.90:
        return "Round"
    elif ratio < 0.75:
        return "Oval"
    return "Square"


def recommend_glass_shape(face_shape):
    return "Round" if face_shape == "Oval" else "Square"


def change_lip_color(frame, color_name, mouth_pts):
    """Approximate lip region using YuNet mouth corner landmarks."""
    color_map = {
        "classic_red": (0, 0, 255),   "deep_red":    (0, 0, 139),
        "cherry_red":  (0, 0, 205),   "rose_red":    (0, 102, 204),
        "wine_red":    (0, 0, 128),   "brick_red":   (0, 64, 128),
        "coral_red":   (0, 128, 255), "berry_red":   (0, 0, 153),
        "ruby_red":    (0, 17, 255),  "crimson_red": (60, 20, 220),
    }
    color = color_map.get(color_name)
    if color is None or mouth_pts is None:
        return frame

    lx, ly = mouth_pts[0]
    rx, ry = mouth_pts[1]
    cx = int((lx + rx) / 2)
    cy = int((ly + ry) / 2) + 4
    lip_w = max(int(np.linalg.norm([rx - lx, ry - ly]) * 0.58), 5)
    lip_h = max(int(lip_w * 0.32), 4)

    # Two ellipses — upper and lower lip
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy - lip_h // 3), (lip_w, lip_h // 2), 0, 180, 360, 255, -1)
    cv2.ellipse(mask, (cx, cy + lip_h // 3), (lip_w, lip_h // 2), 0, 0, 180, 255, -1)

    colored = np.full_like(frame, color)
    alpha_lip = 0.55
    lip_area = cv2.bitwise_and(colored, colored, mask=mask)
    orig_area = cv2.bitwise_and(frame, frame, mask=mask)
    blended = cv2.addWeighted(lip_area, alpha_lip, orig_area, 1 - alpha_lip, 0)
    frame = frame.copy()
    frame[mask == 255] = blended[mask == 255]
    return frame


# ── Main processing ───────────────────────────────────────────────────────────
def process_frame(frame):
    global overlay
    frame = np.array(frame, copy=True)   # RGB from Gradio
    h, w = frame.shape[:2]

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame_bgr)

    face_shape, glass_shape = "Unknown", "Unknown"
    mouth_pts = None

    if faces is not None:
        for face in faces:
            fx, fy, fw, fh = face[:4].astype(int)
            pts = face[4:14].reshape(5, 2).astype(int)
            lx, ly = pts[0]
            rx, ry = pts[1]
            cx, cy = (lx + rx) // 2, (ly + ry) // 2
            angle  = -np.degrees(np.arctan2(ry - ly, rx - lx))
            mouth_pts = (pts[3], pts[4])

            ov = cv2.resize(overlay, (int(fw * 1.15), int(fh * 0.8)))
            M  = cv2.getRotationMatrix2D((ov.shape[1] // 2, ov.shape[0] // 2), angle, 1.0)
            ov = cv2.warpAffine(ov, M, (ov.shape[1], ov.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            try:
                frame = overlay_png(frame, ov, [cx - ov.shape[1] // 2, cy - ov.shape[0] // 2])
            except Exception as e:
                print(f"Overlay error: {e}")

            face_shape  = determine_face_shape(fw, fh)
            glass_shape = recommend_glass_shape(face_shape)

    return frame, face_shape, glass_shape, mouth_pts


def apply_filter(frame, transform):
    if transform == "cartoon":
        img = cv2.pyrUp(cv2.pyrUp(cv2.pyrDown(cv2.pyrDown(frame))))
        for _ in range(6):
            img = cv2.bilateralFilter(img, 9, 9, 7)
        edges = cv2.adaptiveThreshold(
            cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 7),
            255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        return cv2.bitwise_and(img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    elif transform == "edges":
        return cv2.cvtColor(cv2.Canny(frame, 100, 200), cv2.COLOR_GRAY2BGR)
    elif transform == "sepia":
        k = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
        return cv2.cvtColor(np.clip(cv2.transform(frame, k), 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    elif transform == "negative":
        return cv2.bitwise_not(frame)
    elif transform == "sketch":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_blur = cv2.bitwise_not(cv2.GaussianBlur(cv2.bitwise_not(gray), (21, 21), 0))
        return cv2.cvtColor(cv2.divide(gray, inv_blur, scale=256.0), cv2.COLOR_GRAY2BGR)
    elif transform == "blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    return frame


def save_frame(frame):
    if frame is None:
        return None
    path = os.path.join(tempfile.gettempdir(),
                        f"glass_tryon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    return path


def webcam_input(frame, transform, lip_color):
    if frame is None:
        return None, "", ""
    frame, face_shape, glass_shape, mouth_pts = process_frame(frame)
    if transform != "none" and lip_color == "none":
        frame = apply_filter(frame, transform)
    elif lip_color != "none" and transform == "none":
        frame = change_lip_color(frame, lip_color, mouth_pts)
    return frame, face_shape, glass_shape


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align:center;font-weight:bold;'>🤓 Glasses Virtual Try-On 🕶️👓</h1>")
    with gr.Column():
        with gr.Group():
            gr.Markdown("<p style='color:purple;'>🟣Only one filter can be applied at a time.</p>")
            with gr.Row():
                transform = gr.Dropdown(
                    choices=["cartoon", "edges", "sepia", "negative", "sketch", "blur", "none"],
                    value="none", label="Select Filter")
                lip_color = gr.Dropdown(
                    choices=["classic_red", "deep_red", "cherry_red", "rose_red", "wine_red",
                             "brick_red", "coral_red", "berry_red", "ruby_red", "crimson_red", "none"],
                    value="none", label="Select Lip Color")
            gr.Markdown("<p style='color:purple;'>🟣Start the webcam on the left — processed output appears on the right.</p>")
            with gr.Row():
                input_img  = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam")
                output_img = gr.Image(label="Output")
            next_button = gr.Button("Next Glasses ➡️")
            gr.Markdown("<p style='color:purple;'>🟣Detected Face Shape and Recommended Glass Shape</p>")
            with gr.Row():
                face_shape_out  = gr.Textbox(label="Detected Face Shape")
                glass_shape_out = gr.Textbox(label="Recommended Glass Shape")
            save_button   = gr.Button("Save as Picture 📌")
            download_link = gr.File(label="Download Saved Picture")

    input_img.stream(webcam_input,
                     [input_img, transform, lip_color],
                     [output_img, face_shape_out, glass_shape_out],
                     stream_every=0.1)
    next_button.click(change_glasses, [], [])
    save_button.click(save_frame, [output_img], [download_link])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"))
