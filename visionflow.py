import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Initialize OCR
reader = easyocr.Reader(['en'])

# Load general YOLO model (for cars, trucks, etc.)
model = YOLO('yolov5s.pt')

# Load video
video_path = 'traffic_video2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print('Error: Could not open video file.')
    exit()

cap.set(cv2.CAP_PROP_FPS, 15)

# Plate color check function
def check_plate_color(plate_image):
    hsv_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])

    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

    if np.sum(yellow_mask) > np.sum(white_mask):
        return 'Yellow'
    elif np.sum(white_mask) > np.sum(yellow_mask):
        return 'White'
    else:
        return 'Unknown'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 inference
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]

            # Check for vehicle classes only (car, truck, bus, etc.)
            if conf > 0.4 and label in ['car', 'truck', 'bus', 'motorbike']:
                # Crop vehicle region
                vehicle_crop = frame[y1:y2, x1:x2]

                # OCR on the vehicle crop
                ocr_results = reader.readtext(vehicle_crop)
                for (bbox, text, prob) in ocr_results:
                    if prob > 0.5:
                        plate_text = text

                        # Extract bounding box inside vehicle crop (relative)
                        (tl, tr, br, bl) = bbox
                        tl = tuple(map(int, tl))
                        br = tuple(map(int, br))

                        # Get the cropped license plate
                        plate_crop = vehicle_crop[tl[1]:br[1], tl[0]:br[0]]

                        # Detect plate color
                        plate_color = check_plate_color(plate_crop) if plate_crop.size > 0 else 'Unknown'

                        print(f"License Plate: {plate_text}")
                        print(f"Plate Color: {plate_color}")

                        if plate_color in ['Yellow', 'Red']:
                            print("Priority: Emergency Vehicle (Yellow/Red Plate)")
                        else:
                            print("Normal Traffic")

                        # Annotate license plate text
                        cv2.putText(frame, f'{plate_text} ({plate_color})', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw bounding box around vehicle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()