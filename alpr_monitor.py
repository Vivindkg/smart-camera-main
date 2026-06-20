import cv2
import easyocr
import numpy as np
import time
import argparse
import os
from ultralytics import YOLO

def preprocess_plate(plate_img):
    """
    Apply image processing to the license plate to improve OCR accuracy.
    """
    # Resize to make text larger
    plate_img = cv2.resize(plate_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter to reduce noise while keeping edges sharp
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    # Apply adaptive threshold to handle different lighting
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return gray, thresh

def main():
    parser = argparse.ArgumentParser(description="Automatic License Plate Recognition (ALPR)")
    parser.add_argument('--video', type=str, default='0', help="Path to video file or '0' for webcam")
    parser.add_argument('--log', type=str, default='license_plates.log', help="File to save the read plates")
    args = parser.parse_args()

    print("Loading AI Models... This may take a moment.")
    
    # 1. Load YOLOv8 for vehicle detection
    model = YOLO('yolov8s.pt')
    # COCO classes for vehicles: 2: car, 3: motorcycle, 5: bus, 7: truck
    VEHICLE_CLASSES = [2, 3, 5, 7]

    # 2. Load Haar Cascade for license plate localization
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    if plate_cascade.empty():
        print("Error: Could not load the plate cascade classifier.")
        return

    # 3. Load EasyOCR for text recognition
    print("Initializing EasyOCR. This will download models if running for the first time...")
    reader = easyocr.Reader(['en'], gpu=True) # Will automatically fall back to CPU if no GPU

    print("Models loaded successfully!")

    video_source = int(args.video) if args.video.isdigit() else args.video
    is_image = isinstance(video_source, str) and video_source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

    if not is_image:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{args.video}'")
            return

    # To prevent spamming the log file with the same plate over and over
    seen_plates = set()
    log_file = open(args.log, 'a')
    
    print("\nStarting ALPR system. Press 'q' to quit.\n")

    # Limit OCR frequency to avoid massive CPU lag
    frame_count = 0
    
    while True:
        if is_image:
            frame = cv2.imread(video_source)
            if frame is None:
                print(f"Error: Could not read image '{video_source}'")
                break
        else:
            ret, frame = cap.read()
            if not ret:
                break
            
        frame_count += 1
        
        # Run vehicle detection
        results = model(frame, verbose=False, conf=0.4)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    # Vehicle found!
                    vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
                    cv2.putText(frame, "Vehicle", (vx1, vy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Extract the vehicle region to search for plates
                    vehicle_roi = frame[vy1:vy2, vx1:vx2]
                    if vehicle_roi.size == 0:
                        continue
                        
                    # Find plates within the vehicle using Haar Cascade
                    gray_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                    plates = plate_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=4, minSize=(30, 10))
                    
                    for (px, py, pw, ph) in plates:
                        # Absolute coordinates on the main frame
                        abs_px1 = vx1 + px
                        abs_py1 = vy1 + py
                        abs_px2 = abs_px1 + pw
                        abs_py2 = abs_py1 + ph
                        
                        cv2.rectangle(frame, (abs_px1, abs_py1), (abs_px2, abs_py2), (0, 255, 0), 2)
                        
                        # Only run heavy OCR every few frames, but always run it for static images
                        if is_image or frame_count % 3 == 0:
                            plate_img = frame[abs_py1:abs_py2, abs_px1:abs_px2]
                            gray_plate, thresh_plate = preprocess_plate(plate_img)
                            
                            # Read text
                            text_results = reader.readtext(gray_plate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
                            
                            if len(text_results) > 0:
                                # Get the text with the highest confidence
                                text_results.sort(key=lambda x: x[2], reverse=True)
                                best_text = text_results[0][1]
                                confidence = text_results[0][2]
                                
                                # Filter out garbage reads
                                if len(best_text) >= 4 and confidence > 0.4:
                                    # Draw text
                                    cv2.putText(frame, f"{best_text} ({confidence:.2f})", (abs_px1, abs_py1 - 10), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                                    
                                    # Log it
                                    if best_text not in seen_plates:
                                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                        log_entry = f"[{timestamp}] Plate: {best_text} | Conf: {confidence:.2f}\n"
                                        print(log_entry.strip())
                                        log_file.write(log_entry)
                                        log_file.flush()
                                        seen_plates.add(best_text)

        # UI
        cv2.putText(frame, "ALPR System Active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Logged Plates: {len(seen_plates)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display
        display_frame = frame
        if frame.shape[1] > 1200:
            display_frame = cv2.resize(frame, (1200, int(1200 * frame.shape[0] / frame.shape[1])))
            
        cv2.imshow('Automatic License Plate Recognition', display_frame)

        if is_image:
            print("Image processed! Press any key in the image window to exit.")
            cv2.waitKey(0)
            break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if not is_image:
        cap.release()
    log_file.close()
    cv2.destroyAllWindows()
    print(f"\nSaved all detected plates to '{args.log}'")

if __name__ == "__main__":
    main()
