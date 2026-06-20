import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from collections import defaultdict
import time

def main():
    parser = argparse.ArgumentParser(description="Traffic Monitor: Count cars upwards and detect lane changes")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='traffic_output.mp4', help='Path to save output video')
    args = parser.parse_args()

    print("Loading YOLOv8 model...")
    # Using the small model which is already downloaded and is a good balance of speed and accuracy
    model = YOLO('yolov8s.pt') 
    
    # We will track vehicles only. COCO classes: car (2), motorcycle (3), bus (5), truck (7)
    vehicle_classes = [2, 3, 5, 7]

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

    # Tracking state
    track_history = defaultdict(list)
    upward_count = 0
    counted_ids = set()
    lane_changers = set()
    
    # Counting line configuration (horizontal line at 50% height from the top)
    COUNTING_LINE_Y = int(height * 0.5)
    
    # Lane divider configuration
    # Defining a single vertical line down the center of the screen
    LANE_LINES_X = [int(width * 0.5)]
    HISTORY_LENGTH = 10 # Short history just for drawing the blue trailing tail

    print(f"Processing video '{args.video}'...")
    
    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLOv8 tracking, persisting tracks between frames
        results = model.track(frame, persist=True, classes=vehicle_classes, verbose=False)
        
        # Draw the counting line
        cv2.line(frame, (0, COUNTING_LINE_Y), (width, COUNTING_LINE_Y), (0, 255, 255), 2)
        cv2.putText(frame, "Counting Line", (10, COUNTING_LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw the lane dividers (from the counting line down to the bottom)
        for lx in LANE_LINES_X:
            cv2.line(frame, (lx, COUNTING_LINE_Y), (lx, height), (255, 255, 0), 2)
            cv2.putText(frame, "Lane", (lx + 5, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                history = track_history[track_id]
                history.append((cx, cy))
                
                # Keep history length bounded
                if len(history) > HISTORY_LENGTH:
                    history.pop(0)
                    
                is_lane_changing = track_id in lane_changers
                
                # Logic for Counting Upward
                if len(history) >= 2:
                    prev_cx, prev_cy = history[-2]
                    curr_cx, curr_cy = history[-1]
                    
                    # If the center crosses the line from bottom (greater Y) to top (smaller Y)
                    if prev_cy > COUNTING_LINE_Y and curr_cy <= COUNTING_LINE_Y:
                        if track_id not in counted_ids:
                            upward_count += 1
                            counted_ids.add(track_id)
                
                # Logic for Lane Changing
                # If the car crosses any of our defined vertical lane dividers, it changed lanes
                if not is_lane_changing and len(history) >= 2:
                    prev_cx = history[-2][0]
                    curr_cx = history[-1][0]
                    
                    for lx in LANE_LINES_X:
                        # Only consider a lane change if it happens on the bottom half of the screen (where the line is drawn)
                        if curr_cy > COUNTING_LINE_Y or prev_cy > COUNTING_LINE_Y:
                            # Check if the car's bounding box is touching/driving on the line
                            if x1 <= lx <= x2:
                                is_lane_changing = True
                                lane_changers.add(track_id)
                                break
                
                # Draw Box and Info
                color = (0, 0, 255) if is_lane_changing else (0, 255, 0)
                label = f"ID: {track_id}"
                if is_lane_changing:
                    label += " - LANE CHANGE"
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw history track (tail)
                points = np.array(history).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        # Draw overall count
        cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Upward Count: {upward_count}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        # Display
        display_frame = frame
        if width > 1000:
            display_frame = cv2.resize(frame, (1000, int(1000 * height / width)))
        cv2.imshow('Traffic Monitor', display_frame)
        
        loop_time = time.time() - loop_start
        wait_ms = max(1, int(1000 / fps) - int(loop_time * 1000))
        
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete! Output saved to '{args.output}'")

if __name__ == "__main__":
    main()
