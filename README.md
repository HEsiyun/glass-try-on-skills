# 👓 Glasses Virtual Try-On System

A real-time computer vision application for virtual glasses try-on using **OpenCV + YuNet + Gradio**, designed for low-latency interactive webcam experiences.

![CV](https://img.shields.io/badge/CV-OpenCV-green)
![Model](https://img.shields.io/badge/Model-YuNet%20ONNX-blue)
![UI](https://img.shields.io/badge/UI-Gradio-orange)
![Language](https://img.shields.io/badge/Python-3.10-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## ✨ Overview

This project implements a **real-time virtual try-on system** that overlays glasses onto a user's face using webcam input.

It combines:

- Lightweight **face detection (YuNet)**
- Geometric **landmark-based alignment**
- **Image compositing**
- **Interactive UI (Gradio)**

The system is optimized for:

> ⚡ Low latency · 🎯 Acceptable alignment accuracy · 🧠 Simplicity & deployability

---

## 🧠 Key Features

- 🎥 **Real-time Webcam Processing**
- 🧍 **Face Detection (YuNet ONNX model)**
- 👁️ **Eye Landmark-Based Alignment**
- 🕶️ **Dynamic Glasses Overlay**
- 🎨 **Image Filters** — cartoon, sketch, sepia, etc.
- 💾 **Save Output Image**
- 🔄 **Switch Between Multiple Glasses Styles**

---

## ⚙️ System Architecture

```
Gradio Interface
       ↓
Frame Processing (RGB → BGR)
       ↓
YuNet Face Detection (ONNX)
       ↓
5-Point Landmark Extraction
       ↓
Geometry Estimation (center + angle)
       ↓
Glasses Transformation (scale + rotate)
       ↓
Alpha Blending Overlay
       ↓
Optional Filters
       ↓
Output Display
```

---

## ⚖️ Design Considerations

### Why YuNet (instead of MediaPipe)?

- Faster inference (real-time friendly)
- Lightweight ONNX model
- Sufficient 5-point landmarks for this use case

### Trade-offs

| Factor     | Choice   | Reason                |
| ---------- | -------- | --------------------- |
| Accuracy   | Medium   | 5-point landmarks     |
| Latency    | Low ⚡   | Real-time constraint  |
| Complexity | Low      | Single-model pipeline |

---

## 📁 Project Structure

```
glass-try-on/
├── app.py
├── requirements.txt
├── packages.txt
├── face_detection_yunet_2023mar.onnx
├── glasses/
│   ├── glass1.png
│   ├── glass2.png
│   └── …
└── README.md
```

---

## 🚀 Local Setup

### 1. Clone Repo

```bash
git clone https://github.com/YOUR_USERNAME/glass-try-on.git
cd glass-try-on
```

### 2. Create Environment

**Conda (recommended)**

```bash
conda create -n glass-try-on python=3.10 -y
conda activate glass-try-on
```

**or venv**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Packages (Linux only)

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

---

## ▶️ Run Locally

```bash
python app.py
```

Then open: http://127.0.0.1:7860

> 👉 Allow webcam access when prompted.

---

## 🧪 Quick Check (Optional)

```python
python -c "
import cv2
cv2.FaceDetectorYN.create('face_detection_yunet_2023mar.onnx', '', (320,320))
print('Model loaded successfully')
"
```

---

## ⚠️ Notes

- Input images from Gradio are **RGB**, OpenCV expects **BGR**
- Glasses must be **RGBA PNG** (transparent background)
- Naming convention: `glass1.png`, `glass2.png`, ...

---

## 🚧 Limitations

- Face shape detection is heuristic
- Alignment accuracy limited by 5 landmarks
- No occlusion handling (hair, hands, etc.)
- Webcam performance varies by browser

---

## 🔮 Future Improvements

- More accurate landmark detection (MediaPipe / Dlib)
- Better geometric fitting
- Image upload mode
- Mobile optimization
- Modular pipeline refactor

---

## 📌 Project Highlights (for interviews)

- Real-time CV pipeline design
- Latency vs. accuracy trade-off
- Geometric transformation & overlay
- Lightweight model deployment (ONNX)

---

## 📜 License

MIT