import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import time
import random
from google.colab.patches import cv2_imshow

reader = easyocr.Reader(['en'])
model = YOLO('yolov5su.pt')

video_path = 'traffic_video2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print('Error: Could not open video file.')
    exit()

cap.set(cv2.CAP_PROP_FPS, 30)  # Try forcing a higher FPS

# Plate color check
def check_plate_color(plate_image):
    hsv_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_image, (20, 100, 100), (30, 255, 255))
    white_mask = cv2.inRange(hsv_image, (0, 0, 200), (180, 30, 255))
    if np.sum(yellow_mask) > np.sum(white_mask):
        return 'Yellow'
    elif np.sum(white_mask) > np.sum(yellow_mask):
        return 'White'
    else:
        return 'Unknown'

# Traffic signal logic
lanes = ['R1', 'R2', 'R3', 'R4']
vehicle_counts = {lane: 0 for lane in lanes}
last_green_time = {lane: 0 for lane in lanes}
current_green_lane = random.choice(lanes)
last_switch_time = time.time()
green_min_time = 10  # seconds
vehicle_threshold = 40
time_threshold = 20

def update_signal():
    global current_green_lane, last_switch_time
    now = time.time()

    if now - last_switch_time < green_min_time:
        return  # Enforce minimum green time

    # Decision logic
    max_lane = max(vehicle_counts, key=vehicle_counts.get)
    should_switch = False

    # Condition 1: lane exceeds its threshold
    if vehicle_counts[max_lane] > vehicle_threshold:
        should_switch = True

    # Condition 2: significantly more vehicles than others
    for lane, count in vehicle_counts.items():
        if lane != max_lane and vehicle_counts[max_lane] - count > 15:
            should_switch = True
            break

    # Condition 3: current red lane waited too long
    for lane in lanes:
        if lane != current_green_lane and (now - last_green_time[lane]) > time_threshold:
            max_lane = lane
            should_switch = True
            break

    if should_switch and max_lane != current_green_lane:
        last_green_time[current_green_lane] = now
        current_green_lane = max_lane
        last_switch_time = now

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detected_lanes = {lane: 0 for lane in lanes}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]

            if conf > 0.4 and label in ['car', 'truck', 'bus', 'motorbike']:
                vehicle_crop = frame[y1:y2, x1:x2]
                ocr_results = reader.readtext(vehicle_crop)

                for (bbox, text, prob) in ocr_results:
                    if prob > 0.5:
                        (tl, tr, br, bl) = bbox
                        tl = tuple(map(int, tl))
                        br = tuple(map(int, br))
                        plate_crop = vehicle_crop[tl[1]:br[1], tl[0]:br[0]]
                        plate_color = check_plate_color(plate_crop) if plate_crop.size > 0 else 'Unknown'

                        # Dummy lane mapping logic (for demo purposes)
                        lane = random.choice(lanes)
                        detected_lanes[lane] += 1

                        cv2.putText(frame, f'{text} ({plate_color})', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Update vehicle counts
    for lane in lanes:
        vehicle_counts[lane] += detected_lanes[lane]

    update_signal()

    # Show traffic light signals on frame
    for i, lane in enumerate(lanes):
        color = (0, 255, 0) if lane == current_green_lane else (0, 0, 255)
        signal = '🟢' if lane == current_green_lane else '🔴'
        cv2.putText(frame, f'{lane}: {signal}', (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2_imshow( frame)

    # Fast playback
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
