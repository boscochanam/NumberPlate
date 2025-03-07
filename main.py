from datetime import datetime
import os
from ultralytics import YOLO
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from threading import Event
from loguru import logger
import argparse

# Add argument parser
parser = argparse.ArgumentParser(description='Number Plate Recognition with OCR Engine Selection')
parser.add_argument('--ocr', type=str, default='paddle', choices=['paddle', 'tesseract', 'easyocr'],
                    help='OCR engine to use: paddle, tesseract, easyocr')
args = parser.parse_args()

# Initialize OCR engine based on argument
ocr_engine = args.ocr

if ocr_engine == 'paddle':
    try:
        import paddle
        from paddleocr import PaddleOCR
        if not paddle.is_compiled_with_cuda():
            print("CUDA not available, using CPU for PaddleOCR")
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
        else:
            print("CUDA available, using GPU for PaddleOCR")
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=True)
    except ImportError:
        print("PaddlePaddle not found, please install it.")
        exit()
elif ocr_engine == 'tesseract':
    try:
        import pytesseract
    except ImportError:
        print("Tesseract not found, please install it.  Also make sure you have tesseract installed on your system")
        exit()
elif ocr_engine == 'easyocr':
    try:
        import easyocr
        ocr = easyocr.Reader(['en'])
    except ImportError:
        print("EasyOCR not found, please install it.")
        exit()
    
print(f"Using OCR engine: {ocr_engine}")

# Initialize models
model = YOLO('NumberPlate/license_plate_detector.pt')

# source = "youtube_video.mp4"
# source = "2025-03-05 16-57-54.mp4"
source = "05-03-2025 18_08_04 (UTC+05_30)_003.avi"

# Initialize metrics and display
start_time = time.time()
frame_count = 0
plt.ion()
fig, ax = plt.subplots(figsize=(12, 8))
display_handle = None
stop_event = Event()
paused = False  # Add a paused state
current_frame = None
video_capture = None
frame_delay = 0.01  # Delay in seconds for 1 FPS

# Add variables for recording
recording = False
video_writer = None
recording_start_time = None

def on_key_press(event):
    global paused, current_frame, recording, video_writer, recording_start_time
    
    if event.key == 'q':
        stop_event.set()
    elif event.key == 'right':
        paused = True
        current_frame = get_next_frame()
    elif event.key == 'left':
        paused = True
        current_frame = get_previous_frame()
    elif event.key == 'r':  # Toggle recording with 'r' key
        if not recording:
            # Start recording
            start_recording()
        else:
            # Stop recording
            stop_recording()

def start_recording():
    global recording, video_writer, recording_start_time
    
    if recording:
        return  # Already recording
    
    # Create output directory if it doesn't exist
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    
    # Generate output filename using current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join('recordings', f'recording_{timestamp}.mp4')
    
    # Get video properties
    if video_capture is not None:
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Target FPS for the output video
    else:
        # Fallback to current_frame dimensions if available
        if current_frame is not None:
            height, width = current_frame.shape[:2]
            fps = 30
        else:
            # Default values if no frame is available yet
            width, height = 1280, 720
            fps = 30
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if video_writer.isOpened():
        recording = True
        recording_start_time = time.time()
        print(f"Started recording to {output_path}")
    else:
        print("Failed to initialize video writer")

def stop_recording():
    global recording, video_writer
    
    if recording and video_writer is not None:
        recording_duration = time.time() - recording_start_time
        video_writer.release()
        video_writer = None
        recording = False
        print(f"Recording stopped. Duration: {recording_duration:.2f} seconds")

# Connect key press event
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Create screenshots directory
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

def get_next_frame():
    global video_capture
    success, frame = video_capture.read()
    if success:
        return frame
    else:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = video_capture.read()
        return frame

def get_previous_frame():
    global video_capture
    current_frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
    if current_frame_number > 1:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number - 2)
        success, frame = video_capture.read()
        return frame
    else:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = video_capture.read()
        return frame

try:
    video_capture = cv2.VideoCapture(source)
    while True:
        if stop_event.is_set():
            break

        if not paused:
            success, frame = video_capture.read()
            if not success:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = video_capture.read()
                if not success:
                    print("Failed to read the video stream.")
                    break
            current_frame = frame
        
        result = model(current_frame, stream=False, verbose=False)
        
        # Create a copy for display (with indicators)
        display_frame = current_frame.copy()
        
        # Process detections
        plates_found = 0  # Counter for multiple plates in same frame
        
        for box in result[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            if conf > 0.3:
                plates_found += 1  # Increment plate counter
                # Draw bounding box on both frames
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                plate_region = current_frame[y1:y2, x1:x2]
                
                # Add plate number label to the bounding box (only on display frame)
                cv2.putText(display_frame, f"Plate #{plates_found}", (x1, y1-30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # OCR detection
                try:
                    if ocr_engine == 'paddle':
                        result_ocr = ocr.ocr(plate_region, cls=True)
                        plate_text = ''
                        confidence = 0
                        
                        if result_ocr and result_ocr[0]:  # Check if any text detected
                            logger.info(f"Plate #{plates_found} OCR result: {result_ocr}")
                            
                            # PaddleOCR returns a list of results for each image
                            # Each result is a list of text detections [points, [text, confidence]]
                            best_confidence = 0
                            for line in result_ocr[0]:
                                text = line[1][0]  # Get text
                                conf = line[1][1]  # Get confidence score
                                
                                # Take result with highest confidence
                                if conf > best_confidence:
                                    plate_text = text
                                    confidence = conf
                                    best_confidence = conf
                    elif ocr_engine == 'tesseract':
                        plate_text = pytesseract.image_to_string(plate_region, config='--psm 6')
                        confidence = 0.99 # Assign a default high confidence
                    elif ocr_engine == 'easyocr':
                        result_ocr = ocr.readtext(plate_region)
                        if result_ocr:
                            plate_text = result_ocr[0][1]
                            confidence = result_ocr[0][2]
                        else:
                            plate_text = ''
                            confidence = 0
                    
                    if plate_text and confidence > 0.92 and len(plate_text) >= 9:  # Confidence threshold for OCR
                        # Draw OCR results on both frames, but with different styling
                        # Add to recording frame with minimal styling
                        cv2.putText(current_frame, f"{plate_text}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Add more detailed info to display frame
                        cv2.putText(display_frame, f"#{plates_found}: {plate_text}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        print(f"Detected plate #{plates_found}: {plate_text} (conf: {confidence:.2f})")
                        
                        # Save frame with detection
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        save_path = os.path.join('screenshots', 
                                               f'plate{plates_found}_{timestamp}_{plate_text}.jpg')
                        
                        # Save both the full frame and the isolated plate region
                        cv2.imwrite(save_path, current_frame)
                        plate_only_path = os.path.join('screenshots', 
                                                     f'plate{plates_found}_{timestamp}_{plate_text}_region.jpg')
                        cv2.imwrite(plate_only_path, plate_region)
                except Exception as ocr_error:
                    print(f"OCR Error on plate #{plates_found}: {ocr_error}")
        
        # Display total plates found in the frame (only on display frame)
        if plates_found > 0:
            cv2.putText(display_frame, f"Total Plates: {plates_found}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add recording indicator to display frame (but not to the recorded frame)
        if recording:
            # Calculate recording duration
            rec_duration = time.time() - recording_start_time
            
            # Get frame dimensions
            display_height, display_width = display_frame.shape[:2]
            
            # Red circle and REC text
            cv2.circle(display_frame, (display_width - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_frame, f"REC {rec_duration:.1f}s", (display_width - 120, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write the frame without indicators to the video file
            if video_writer is not None and video_writer.isOpened():
                video_writer.write(current_frame)
        
        # Display using matplotlib (show the display_frame with indicators)
        if display_handle is None:
            display_handle = ax.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        else:
            display_handle.set_data(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        
        plt.pause(0.001)
        if paused:
            paused = False
        else:
            time.sleep(frame_delay)  # Introduce delay to control frame rate

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Make sure to release the video writer
    if recording and video_writer is not None:
        stop_recording()
    
    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()
    plt.close()
    print("Detection stopped")