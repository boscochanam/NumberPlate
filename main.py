from datetime import datetime
import os
from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR
import numpy as np
import time
import matplotlib.pyplot as plt
from threading import Event

# Initialize models
model = YOLO('license_plate_detector.pt')
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
source = "https://youtu.be/dyf5c1iX5Xc"

# Initialize metrics and display
start_time = time.time()
frame_count = 0
plt.ion()
fig, ax = plt.subplots(figsize=(12, 8))
display_handle = None
stop_event = Event()

def on_key_press(event):
    if event.key == 'q':
        stop_event.set()

# Connect key press event
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Create screenshots directory
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

try:
    for result in model(source, stream=True):
        if stop_event.is_set():
            break
            
        frame = result.orig_img
        
        # Process detections
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if conf > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plate_region = frame[y1:y2, x1:x2]
                
                # PaddleOCR detection
                result = ocr.ocr(plate_region, cls=True)
                
                if result[0]:  # Check if any text detected
                    plate_text = result[0][0][1][0]  # Get text with highest confidence
                    confidence = result[0][0][1][1]  # Get confidence score
                    
                    if confidence > 0.5:  # Confidence threshold for OCR
                        cv2.putText(frame, f"Plate: {plate_text}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print(f"Detected plate: {plate_text} (conf: {confidence:.2f})")
                        
                        # Save frame with detection
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        save_path = os.path.join('screenshots', 
                                               f'plate_{timestamp}_{plate_text}.jpg')
                        cv2.imwrite(save_path, frame)
        
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display using matplotlib
        if display_handle is None:
            display_handle = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        else:
            display_handle.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        plt.pause(0.001)

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    plt.close()
    print("Detection stopped")