import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import time
import random
import csv
import os
from google.colab.patches import cv2_imshow

reader = easyocr.Reader(['en'])
model = YOLO('yolov5su.pt')

video_path = 'traffic_video2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print('Error: Could not open video file.')
    exit()

cap.set(cv2.CAP_PROP_FPS, 30)

csv_filename = 'TrafficRecords.csv'
frames_dir = 'frames'

os.makedirs(frames_dir, exist_ok=True)
file_exists = os.path.exists(csv_filename)

# Create CSV file if it doesn't exist
with open(csv_filename, mode='a', newline='') as csv_file:
    fieldnames = ['Timestamp', 'LicensePlate', 'PlateColor', 'Lane', 'VehicleType', 'Confidence', 'FrameImage']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

# Plate color detection
def check_plate_color(plate_image):
    hsv_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv_image, (20, 100, 100), (30, 255, 255))
    white_mask = cv2.inRange(hsv_image, (0, 0, 200), (180, 30, 255))
    lower_red_mask = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
    upper_red_mask = cv2.inRange(hsv_image, (160, 100, 100), (179, 255, 255))
    red_mask = lower_red_mask + upper_red_mask
    yellow_sum = np.sum(yellow_mask)
    white_sum = np.sum(white_mask)
    red_sum = np.sum(red_mask)
    if red_sum > yellow_sum and red_sum > white_sum:
        return 'Red'
    elif yellow_sum > white_sum:
        return 'Yellow'
    elif white_sum > yellow_sum:
        return 'White'
    else:
        return 'Unknown'

# Traffic logic
lanes = ['R1', 'R2', 'R3', 'R4']
vehicle_counts = {lane: 0 for lane in lanes}
last_green_time = {lane: 0 for lane in lanes}
current_green_lane = random.choice(lanes)
last_switch_time = time.time()
green_min_time = 10
vehicle_threshold = 40
time_threshold = 20

def update_signal():
    global current_green_lane, last_switch_time
    now = time.time()
    if now - last_switch_time < green_min_time:
        return

    max_lane = max(vehicle_counts, key=vehicle_counts.get)
    should_switch = False

    if vehicle_counts[max_lane] > vehicle_threshold:
        should_switch = True

    for lane, count in vehicle_counts.items():
        if lane != max_lane and vehicle_counts[max_lane] - count > 15:
            should_switch = True
            break

    for lane in lanes:
        if lane != current_green_lane and (now - last_green_time[lane]) > time_threshold:
            max_lane = lane
            should_switch = True
            break

    if should_switch and max_lane != current_green_lane:
        last_green_time[current_green_lane] = now
        current_green_lane = max_lane
        last_switch_time = now

# Main loop
frame_id = 0
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

                        lane = random.choice(lanes)
                        detected_lanes[lane] += 1
                        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                        frame_filename = f"{frames_dir}/frame_{timestamp}_{text.replace(' ', '_')}.jpg"
                        cv2.imwrite(frame_filename, frame)

                        # Save record to CSV
                        with open(csv_filename, mode='a', newline='') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=[
                                'Timestamp', 'LicensePlate', 'PlateColor', 'Lane', 'VehicleType', 'Confidence', 'FrameImage'
                            ])
                            writer.writerow({
                                'Timestamp': timestamp.replace("_", " "),
                                'LicensePlate': text,
                                'PlateColor': plate_color,
                                'Lane': lane,
                                'VehicleType': label,
                                'Confidence': round(prob, 2),
                                'FrameImage': frame_filename
                            })

                        cv2.putText(frame, f'{text} ({plate_color})', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for lane in lanes:
        vehicle_counts[lane] += detected_lanes[lane]

    update_signal()

    for i, lane in enumerate(lanes):
        color = (0, 255, 0) if lane == current_green_lane else (0, 0, 255)
        signal = '🟢' if lane == current_green_lane else '🔴'
        cv2.putText(frame, f'{lane}: {signal}', (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2_imshow(frame)
    time.sleep(0.03)
    frame_id += 1

cap.release()
cv2.destroyAllWindows()
