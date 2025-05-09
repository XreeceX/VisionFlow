
# 🚦 Smart Traffic Management System with License Plate Recognition

This project detects vehicles and license plates from traffic videos using **YOLOv5** and **EasyOCR**, identifies the **plate color**, and simulates **smart traffic signals**. Detection data is stored in a CSV file and frames are saved locally.

---

## 📁 Project Structure

```
visionflow/
│
├── visionflow.py              # Main application file
├── ViewCSV.py                 # Optional CSV viewer (not used in this README)
├── traffic_video2.mp4         # Sample traffic video input
├── yolov5su.pt                # YOLOv5 model weights (custom/small)
├── frames/                    # Saved image frames with detections
├── TrafficRecords.csv         # Output file with detection logs
├── requirements.txt           # Required Python dependencies
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/visionflow.git
cd visionflow
```

### 2. Create and Activate a Virtual Environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

Make sure to have Python 3.8 or above.

```bash
pip install -r requirements.txt
```

### 4. Download or Place Required Files

- ✅ Ensure `traffic_video2.mp4` is present in the directory.
- ✅ Place your YOLOv5 model file (`yolov5su.pt`) in the same directory.
- 🚫 No need to create `frames/` or `TrafficRecords.csv` manually; they will be auto-created.

---

## ▶️ Run the Project

```bash
python visionflow.py
```

The program will:
- Process video frames.
- Detect vehicles using YOLO.
- Detect license plate text using EasyOCR.
- Determine plate color (Red, Yellow, White).
- Assign a random lane.
- Save frame and detection data to CSV and local image folder.
- Display live traffic signals on video feed.

---

## 📝 Output

- **TrafficRecords.csv**: Contains all detections.
- **frames/**: Contains saved image frames with annotated plates and vehicle info.

Each row in CSV includes:
- Timestamp
- License Plate Text
- Plate Color
- Lane
- Vehicle Type
- Confidence
- Path to Saved Frame Image

---

## 📦 requirements.txt Sample

```txt
opencv-python
numpy
easyocr
ultralytics
```

---

## ⚠️ Notes

- You must have a GPU for optimal performance (for real-time YOLO + OCR).
- If running on **Google Colab**, replace `cv2.imshow` with `cv2_imshow` (already done in code).
- `yolov5su.pt` is a lightweight YOLO model; you can use your custom-trained one.

---

## 📸 Sample Output Preview

You can check the saved frame images in the `frames/` directory after running.
