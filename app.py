import cv2
import numpy as np
import os
import tempfile
import gradio as gr
import mediapipe as mp
from datetime import datetime

# Face detector
model_path = 'face_detection_yunet_2023mar.onnx'
face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))

# Face mesh
from mediapipe.python.solutions.face_mesh import FaceMesh
face_mesh = FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Glasses state
num = 1

def _load_glass(n):
    img = cv2.imread(f'glasses/glass{n}.png', cv2.IMREAD_UNCHANGED)
    b, g, r, a = cv2.split(img)
    return cv2.merge((r, g, b, a))

overlay = _load_glass(num)
total_glass_num = sum(len(f) for _, _, f in os.walk('glasses'))


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


def determine_face_shape(landmarks):
    jaw_width = np.linalg.norm(landmarks[0] - landmarks[16])
    face_height = np.linalg.norm(landmarks[8] - landmarks[27])
    ratio = jaw_width / face_height
    if ratio > 1.5:
        return "Round"
    elif ratio < 1.2:
        return "Oval"
    return "Square"


def recommend_glass_shape(face_shape):
    return "Round" if face_shape == "Oval" else "Square"


def change_glasses():
    global num, overlay
    num = num % total_glass_num + 1
    overlay = _load_glass(num)


def change_lip_color(frame, color_name):
    color_map = {
        'classic_red': (255, 0, 0),   'deep_red':    (139, 0, 0),
        'cherry_red':  (205, 0, 0),   'rose_red':    (204, 102, 0),
        'wine_red':    (128, 0, 0),   'brick_red':   (128, 64, 0),
        'coral_red':   (255, 128, 0), 'berry_red':   (153, 0, 0),
        'ruby_red':    (255, 17, 0),  'crimson_red': (220, 20, 60),
    }
    color = color_map.get(color_name)
    if color is None:
        return frame

    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return frame

    for lm in results.multi_face_landmarks:
        def pt(i):
            return (lm.landmark[i].x * frame.shape[1], lm.landmark[i].y * frame.shape[0])

        upper = np.array([pt(i) for i in [61,185,40,39,37,0,267,269,270,409,291,61]], np.int32)
        lower = np.array([pt(i) for i in [61,146,91,181,84,17,314,405,321,375,291,61]], np.int32)
        teeth = np.array([pt(i) for i in [78,95,88,178,87,14,317,402,318,324,308,78]], np.int32)

        lip_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(lip_mask, [np.concatenate((upper, lower))], 255)
        teeth_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(teeth_mask, [teeth], 255)
        mask = cv2.subtract(lip_mask, teeth_mask)

        colored = np.full_like(frame, color)
        frame = cv2.add(
            cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask)),
            cv2.bitwise_and(colored, colored, mask=mask)
        )
    return frame


def process_frame(frame):
    global overlay
    frame = np.array(frame, copy=True)
    h, w = frame.shape[:2]

    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    face_shape, glass_shape = "Unknown", "Unknown"

    if faces is not None and results.multi_face_landmarks:
        for face in faces:
            x, y, fw, fh = face[:4].astype(int)
            pts = face[4:14].reshape(5, 2).astype(int)
            lx, ly = pts[0]
            rx, ry = pts[1]
            cx, cy = (lx + rx) // 2, (ly + ry) // 2
            angle = -np.degrees(np.arctan2(ry - ly, rx - lx))

            ov = cv2.resize(overlay, (int(fw * 1.15), int(fh * 0.8)))
            M = cv2.getRotationMatrix2D((ov.shape[1] // 2, ov.shape[0] // 2), angle, 1.0)
            ov = cv2.warpAffine(ov, M, (ov.shape[1], ov.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            try:
                frame = overlay_png(frame, ov, [cx - ov.shape[1] // 2, cy - ov.shape[0] // 2])
            except Exception as e:
                print(f"Overlay error: {e}")

            for mp_lm in results.multi_face_landmarks:
                landmarks = np.array([(l.x * w, l.y * h, l.z * w) for l in mp_lm.landmark])
                face_shape = determine_face_shape(landmarks)
                glass_shape = recommend_glass_shape(face_shape)

    return frame, face_shape, glass_shape


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
    path = os.path.join(tempfile.gettempdir(),
                        f"glass_tryon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return path


def webcam_input(frame, transform, lip_color):
    frame, face_shape, glass_shape = process_frame(frame)
    if transform != "none" and lip_color == "none":
        frame = apply_filter(frame, transform)
    elif lip_color != "none" and transform == "none":
        frame = change_lip_color(frame, lip_color)
    return frame, face_shape, glass_shape


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
            gr.Markdown("<p style='color:purple;'>🟣Click the Webcam icon to start, then press record.</p>")
            input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True)
            next_button = gr.Button("Next Glasses ➡️")
            gr.Markdown("<p style='color:purple;'>🟣Detected Face Shape and Recommended Glass Shape</p>")
            with gr.Row():
                face_shape_out  = gr.Textbox(label="Detected Face Shape")
                glass_shape_out = gr.Textbox(label="Recommended Glass Shape")
            save_button   = gr.Button("Save as Picture 📌")
            download_link = gr.File(label="Download Saved Picture")

    input_img.stream(webcam_input,
                     [input_img, transform, lip_color],
                     [input_img, face_shape_out, glass_shape_out],
                     stream_every=0.1)
    next_button.click(change_glasses, [], [])
    save_button.click(save_frame, [input_img], [download_link])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"))
