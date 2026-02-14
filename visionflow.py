"""
VisionFlow - LPR System for Traffic Management.
Detects vehicles and license plates, simulates smart traffic signals.
"""
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# Display: use cv2_imshow in Colab, cv2.imshow locally
try:
    from google.colab.patches import cv2_imshow
    def _display_frame(frame): cv2_imshow(frame)
except ImportError:
    def _display_frame(frame):
        cv2.imshow('VisionFlow', frame)
        cv2.waitKey(1)

# Config
VIDEO_PATH = 'traffic_video2.mp4'
CSV_FILENAME = 'TrafficRecords.csv'
FRAMES_DIR = Path('frames')
FIELDNAMES = ['Timestamp', 'LicensePlate', 'PlateColor', 'Lane', 'VehicleType', 'Confidence', 'FrameImage']

# Traffic signal params
LANES = ['R1', 'R2', 'R3', 'R4']
GREEN_MIN_TIME = 10
VEHICLE_THRESHOLD = 40
TIME_THRESHOLD = 20
LANE_IMBALANCE_THRESHOLD = 15

reader = easyocr.Reader(['en'])
model = YOLO('yolov5su.pt')

FRAMES_DIR.mkdir(exist_ok=True)
file_exists = Path(CSV_FILENAME).exists()
if not file_exists:
    with open(CSV_FILENAME, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def check_plate_color(plate_image: np.ndarray) -> str:
    """Determine plate color from HSV."""
    if plate_image.size == 0:
        return 'Unknown'
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    yellow = np.sum(cv2.inRange(hsv, (20, 100, 100), (30, 255, 255)))
    white = np.sum(cv2.inRange(hsv, (0, 0, 200), (180, 30, 255)))
    red = np.sum(cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))) + \
          np.sum(cv2.inRange(hsv, (160, 100, 100), (179, 255, 255)))
    if red > yellow and red > white:
        return 'Red'
    if yellow > white:
        return 'Yellow'
    if white > yellow:
        return 'White'
    return 'Unknown'


def update_signal(
    vehicle_counts: dict,
    current_green_lane: str,
    last_green_time: dict,
    last_switch_time: float
) -> tuple[str, float, bool]:
    """
    Determine next green lane.
    Returns (new_green_lane, new_last_switch_time, did_switch).
    """
    now = time.time()
    if now - last_switch_time < GREEN_MIN_TIME:
        return current_green_lane, last_switch_time, False

    max_lane = max(vehicle_counts, key=vehicle_counts.get)
    should_switch = False

    if vehicle_counts[max_lane] > VEHICLE_THRESHOLD:
        should_switch = True

    for lane, count in vehicle_counts.items():
        if lane != max_lane and vehicle_counts[max_lane] - count > LANE_IMBALANCE_THRESHOLD:
            should_switch = True
            break

    for lane in LANES:
        if lane != current_green_lane and (now - last_green_time[lane]) > TIME_THRESHOLD:
            max_lane = lane
            should_switch = True
            break

    if should_switch and max_lane != current_green_lane:
        last_green_time[current_green_lane] = now
        vehicle_counts[max_lane] = 0  # Reset queue for newly green lane
        return max_lane, now, True

    return current_green_lane, last_switch_time, False


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print('Error: Could not open video file.')
        return

    vehicle_counts = {lane: 0 for lane in LANES}
    last_green_time = {lane: 0.0 for lane in LANES}
    current_green_lane = LANES[0]
    last_switch_time = time.time()

    # Batch CSV writes - accumulate rows, write periodically
    pending_rows = []
    BATCH_SIZE = 10

    with open(CSV_FILENAME, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            detected_lanes = {lane: 0 for lane in LANES}

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = model.names[cls]

                    if conf <= 0.4 or label not in ('car', 'truck', 'bus', 'motorbike'):
                        continue

                    vehicle_crop = frame[y1:y2, x1:x2]
                    ocr_results = reader.readtext(vehicle_crop)

                    for (bbox, text, prob) in ocr_results:
                        if prob <= 0.5:
                            continue

                        (tl, tr, br, bl) = bbox
                        tl = tuple(map(int, tl))
                        br = tuple(map(int, br))
                        plate_crop = vehicle_crop[
                            max(0, tl[1]):min(vehicle_crop.shape[0], br[1]),
                            max(0, tl[0]):min(vehicle_crop.shape[1], br[0])
                        ]
                        plate_color = check_plate_color(plate_crop)

                        lane = LANES[sum(ord(c) for c in text) % len(LANES)]  # Deterministic lane from plate
                        detected_lanes[lane] += 1

                        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
                        safe_text = text.replace(' ', '_').replace('/', '_')
                        frame_filename = str(FRAMES_DIR / f"frame_{timestamp}_{safe_text}.jpg")
                        cv2.imwrite(frame_filename, frame)

                        pending_rows.append({
                            'Timestamp': timestamp.replace('_', ' '),
                            'LicensePlate': text,
                            'PlateColor': plate_color,
                            'Lane': lane,
                            'VehicleType': label,
                            'Confidence': round(prob, 2),
                            'FrameImage': frame_filename,
                        })

                        cv2.putText(
                            frame, f'{text} ({plate_color})',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        frame, label,
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                    )

            for lane in LANES:
                vehicle_counts[lane] += detected_lanes[lane]

            current_green_lane, last_switch_time, _ = update_signal(
                vehicle_counts, current_green_lane, last_green_time, last_switch_time
            )

            # Flush CSV batch
            if len(pending_rows) >= BATCH_SIZE:
                writer.writerows(pending_rows)
                pending_rows.clear()

            for i, lane in enumerate(LANES):
                color = (0, 255, 0) if lane == current_green_lane else (0, 0, 255)
                signal = '🟢' if lane == current_green_lane else '🔴'
                cv2.putText(
                    frame, f'{lane}: {signal}',
                    (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
                )

            _display_frame(frame)
            time.sleep(0.03)

        # Flush remaining rows
        if pending_rows:
            writer.writerows(pending_rows)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
