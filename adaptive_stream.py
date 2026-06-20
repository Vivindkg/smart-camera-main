import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Adaptive Stream: Dynamically adjust bitrate and resolution based on human presence")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='adaptive_output.mp4', help='Path to save output video')
    args = parser.parse_args()

    print("Loading YOLOv8 model for human detection...")
    model = YOLO('yolov8s.pt') 
    PERSON_CLASS = 0

    is_image = args.video.lower().endswith(('.png', '.jpg', '.jpeg'))
    
    if is_image:
        print("This model is meant for live video streams, but running on image for testing.")
        frame = cv2.imread(args.video)
        if frame is None:
            print(f"Error: Could not read image '{args.video}'")
            return
        width = frame.shape[1]
        height = frame.shape[0]
        fps = 30.0
    else:
        video_source = int(args.video) if args.video.isdigit() else args.video
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{args.video}'")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing '{args.video}'...")
    
    # Adaptive Streaming State
    active_mode = False
    last_human_seen_time = 0
    COOLDOWN_SECONDS = 5.0 # Stay in High Quality for 5 seconds after a person leaves
    
    # Idle settings
    LOW_RES_WIDTH = 320
    LOW_RES_HEIGHT = int(320 * (height / width))
    
    while True:
        loop_start = time.time()
        
        if not is_image:
            ret, frame = cap.read()
            if not ret:
                break
                
        # To simulate the actual AI running on a low-end idle stream, we process the detection on a downscaled frame
        # (This also makes the detection extremely fast when idle)
        detect_frame = cv2.resize(frame, (LOW_RES_WIDTH, LOW_RES_HEIGHT))
        
        # Run YOLOv8 detection
        results = model(detect_frame, classes=[PERSON_CLASS], verbose=False)
        
        human_detected = False
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for conf in results[0].boxes.conf.cpu():
                if conf > 0.4:  # Confidence threshold
                    human_detected = True
                    break
                    
        current_time = time.time()
        if human_detected:
            last_human_seen_time = current_time
            active_mode = True
        elif current_time - last_human_seen_time > COOLDOWN_SECONDS:
            active_mode = False
            
        # Prepare output frame
        output_frame = frame.copy()
        
        if not active_mode:
            # SIMULATE LOW BITRATE/RESOLUTION
            # Downscale drastically, then upscale back to original size to show the pixelation
            pixelated = cv2.resize(output_frame, (LOW_RES_WIDTH, LOW_RES_HEIGHT), interpolation=cv2.INTER_NEAREST)
            output_frame = cv2.resize(pixelated, (width, height), interpolation=cv2.INTER_NEAREST)
            
            status_text = "IDLE - LOW BITRATE STREAM"
            status_color = (0, 0, 255) # Red for idle
            bitrate_text = "~150 kbps"
        else:
            # HIGH QUALITY MODE
            # We use the raw, native 1080p+ frame with no compression
            status_text = "ACTIVE - FULL HD HIGH BITRATE"
            status_color = (0, 255, 0) # Green for active
            bitrate_text = "~6000 kbps"
            
            # Draw bounding boxes (Need to run detection on full frame or scale up boxes)
            full_results = model(output_frame, classes=[PERSON_CLASS], verbose=False)
            if full_results[0].boxes is not None:
                for box, conf in zip(full_results[0].boxes.xyxy.cpu(), full_results[0].boxes.conf.cpu()):
                    if conf > 0.4:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(output_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw UI Overlay
        cv2.rectangle(output_frame, (10, 10), (550, 100), (0, 0, 0), -1)
        cv2.putText(output_frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(output_frame, f"Simulated Data Usage: {bitrate_text}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if is_image:
            cv2.imwrite(args.output.replace('.mp4', '.jpg'), output_frame)
            print("Press any key in the image window to close it...")
            
            display_frame = output_frame
            if width > 1000:
                display_frame = cv2.resize(output_frame, (1000, int(1000 * height / width)))
            cv2.imshow('Adaptive Stream Model', display_frame)
            cv2.waitKey(0)
            break
        else:
            out.write(output_frame)
            
            display_frame = output_frame
            if width > 1000:
                display_frame = cv2.resize(output_frame, (1000, int(1000 * height / width)))
            cv2.imshow('Adaptive Stream Model', display_frame)
            
            loop_time = time.time() - loop_start
            
            # For pre-recorded videos, artificial delays cause slow-motion playback.
            # We just use waitKey(1) to let the window update as fast as possible.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if not is_image:
        cap.release()
        out.release()
        
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print(f"Processing complete! Output saved to '{args.output}'")

if __name__ == "__main__":
    main()
