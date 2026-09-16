# 🚗🚶‍♂️ Vehicle & Person Detection System

A computer vision web application that detects **people, bicycles, cars, motorbikes, buses, and trucks** in uploaded images using the **YOLOv4-tiny** object detection model, built with **OpenCV DNN** and **Streamlit**.

---

## 📌 Project Overview

This project provides an interactive web interface for real-time object detection on static images. Users can upload any image, and the system will identify and draw bounding boxes around six target object classes, along with their confidence scores. Detection sensitivity can be fine-tuned live using adjustable sliders.

---

## ❓ Problem Description

Detecting and classifying vehicles and pedestrians in images has wide applications in traffic monitoring, smart city systems, surveillance, and road safety analysis. Manually reviewing images to identify these objects is slow and error-prone. This project automates that process using a lightweight, fast deep learning model suitable for near real-time inference, even without a GPU.

---

## 📊 Dataset & Model Used

- **Model:** YOLOv4-tiny (pre-trained on the **COCO dataset**)
- **Framework:** OpenCV's `cv2.dnn` module (Darknet backend)
- **Required files:**
  - `yolov4-tiny.weights` — pre-trained model weights
  - `yolov4-tiny.cfg` — model architecture/configuration file
- **Target classes (subset of the 80 COCO classes):**

| Class ID | Label     |
|----------|-----------|
| 0        | Person    |
| 1        | Bicycle   |
| 2        | Car       |
| 3        | Motorbike |
| 5        | Bus       |
| 7        | Truck     |

> Note: The `.weights` and `.cfg` files are not included in this repository due to file size limits. See [How to Run the Project](#-how-to-run-the-project) for download instructions.

---

## 🔄 Workflow / Architecture

1. **Image Upload** — User uploads a `.jpg`, `.jpeg`, or `.png` image via the Streamlit sidebar.
2. **Preprocessing** — Image is converted to RGB and transformed into a blob (`cv2.dnn.blobFromImage`) sized 416×416.
3. **Inference** — The blob is passed through the YOLOv4-tiny network to get raw detection outputs.
4. **Filtering** — Detections are filtered to keep only the 6 target classes above the confidence threshold.
5. **Non-Max Suppression (NMS)** — Overlapping bounding boxes for the same object are removed using `cv2.dnn.NMSBoxes`.
6. **Visualization** — Final bounding boxes and class labels (with confidence scores) are drawn on the image and displayed to the user.

```
Upload Image → Preprocess (Blob) → YOLOv4-tiny Inference →
Filter by Class & Confidence → Apply NMS → Draw Boxes → Display Result
```

---

## 📈 Results & Evaluation

- The model successfully detects and labels the 6 target classes with bounding boxes and confidence scores overlaid on the image.
- Detection quality can be tuned interactively:
  - **Confidence threshold** (default 0.3) — minimum score for a detection to be considered valid.
  - **NMS threshold** (default 0.4) — controls how aggressively overlapping boxes are merged.
- Since YOLOv4-tiny is a lightweight model, it offers fast inference at a slight trade-off in accuracy compared to larger YOLO variants.

---

## 🛠️ Technologies Used

- **Python 3**
- **OpenCV** (`opencv-python`) — model loading & inference via `cv2.dnn`
- **NumPy** — array/image processing
- **Streamlit** — interactive web UI
- **Pillow (PIL)** — image loading and format conversion
- **YOLOv4-tiny (Darknet)** — pre-trained object detection model

---

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/ialshaiban/PythonApplication333.git
cd PythonApplication333
```

### 2. Install dependencies
```bash
pip install streamlit opencv-python numpy pillow
```

### 3. Download the YOLOv4-tiny model files
Place these two files in the project's root folder:
- [`yolov4-tiny.weights`](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
- [`yolov4-tiny.cfg`](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Use the app
- Open the local URL shown in the terminal (usually `http://localhost:8501`).
- Adjust the confidence and NMS thresholds from the sidebar.
- Upload an image to see detection results.

---

## 🚀 Future Improvements

- Upgrade to **YOLOv8** for improved accuracy and speed.
- Add support for **video and real-time webcam** detection.
- Add **object counting** (e.g., total number of cars/persons detected).
- Deploy the app publicly (Streamlit Cloud / Hugging Face Spaces).
- Add multi-language UI support (Arabic/English toggle).
- Export detection results as CSV/JSON reports.

---

## 🎓 SDAIA Academy

This project was developed as part of the **SDAIA Academy** training program.

## 🔗 Repository Link

[https://github.com/ialshaiban/PythonApplication333](https://github.com/ialshaiban/PythonApplication333)

GitHub Repository Link
GitHub Repository Link
GitHub Repository Link
https://github.com/SDAIAAcademy
https://github.com/SDAIAAcademy
https://github.com/SDAIAAcademy
