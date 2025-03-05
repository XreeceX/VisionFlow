import cv2
import numpy as np
import easyocr  # EasyOCR for license plate recognition
from ultralytics import YOLO  # YOLOv5 from the ultralytics package
import matplotlib.pyplot as plt  # For displaying images in Colab

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # English language for OCR

# Load YOLOv5 model (weights .pt file)
model = YOLO('yolov5s.pt')  # Replace with the path to your downloaded yolov5s.pt file

# Load video or use webcam
video_path = 'traffic_video.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if video file is loaded properly
if not cap.isOpened():
    print('Error: Could not open video file.')
    exit()  # Exit if the video file cannot be loaded

# Set the playback speed (FPS). Lowering FPS will slow down the video.
cap.set(cv2.CAP_PROP_FPS, 15)  # Set the FPS to 15 (adjust as needed)

# Function to check license plate color (basic implementation)
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

    # Run YOLOv5 inference on the frame
    results = model(frame)  

    # Draw bounding boxes on detected vehicles and extract license plate
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]  # Get class name
            
            if conf > 0.4:  # Confidence threshold
                vehicle_crop = frame[y1:y2, x1:x2]
                
                # Use EasyOCR to detect text (license plate)
                ocr_results = reader.readtext(vehicle_crop)
                for (bbox, text, prob) in ocr_results:
                    if prob > 0.5:  # Confidence threshold for OCR
                        plate_text = text
                        print(f"License Plate: {plate_text}")
                        
                        # Check plate color
                        plate_color = check_plate_color(vehicle_crop)
                        print(f"Plate Color: {plate_color}")

                        # Implement priority logic based on color
                        if plate_color == 'Yellow' or plate_color == 'Red':
                            print("Priority: Emergency Vehicle (Yellow/Red Plate)")
                        else:
                            print("Normal Traffic")

                        # Draw the detected bounding box and plate info
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{plate_text} {plate_color}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame using Matplotlib (works in Colab)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    plt.imshow(frame_rgb)
    plt.axis('off')  # Turn off axis
    plt.show()
    
    # Wait for a key press to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
