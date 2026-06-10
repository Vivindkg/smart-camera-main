import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import argparse
import math

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_facing_and_sitting(pose_landmarks, person_box, monitor_boxes):
    """
    Heuristic function to determine if a worker is likely sitting and facing a monitor.
    Uses proximity to the detected monitor and bounding box aspect ratio for posture.
    """
    # If no monitors are detected, they aren't facing a computer
    if not monitor_boxes:
        return False
        
    person_width = person_box[2] - person_box[0]
    person_height = person_box[3] - person_box[1]
    
    # Check if the person is standing based on bounding box aspect ratio
    # We use a higher ratio (2.3) so people sitting close to the camera aren't accidentally marked standing
    if person_width > 0:
        aspect_ratio = person_height / person_width
        if aspect_ratio > 2.3:
            return False # They are standing, so likely not working
            
    # Center of the person bounding box
    px_center = (person_box[0] + person_box[2]) / 2
    py_center = (person_box[1] + person_box[3]) / 2
    
    working = False
    
    for mb in monitor_boxes:
        # Center of the monitor bounding box
        mx_center = (mb[0] + mb[2]) / 2
        my_center = (mb[1] + mb[3]) / 2
        
        dist = calculate_distance((px_center, py_center), (mx_center, my_center))
        
        # If person is within a reasonable distance to the monitor
        if dist < (person_width * 3.5): 
            working = True
            break
            
    return working

def main():
    parser = argparse.ArgumentParser(description="Worker Activity Tracker")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file (.mp4)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to save the output video')
    args = parser.parse_args()

    # Initialize YOLOv8
    print("Loading YOLOv8 model...")
    try:
        model = YOLO('yolov8n.pt') # Lightweight model, will download automatically
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Initialize MediaPipe Pose
    print("Loading MediaPipe Pose...")
    try:
        import mediapipe.python.solutions.pose as mp_pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as e:
        print(f"Warning: MediaPipe Pose not available ({e}). Proceeding with YOLO proximity logic only.")
        pose = None

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video}'")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing video '{args.video}'...")
    frame_count = 0
    
    # Optimization variables
    process_every_n_frames = 5 # Run AI every 5th frame
    last_person_boxes = []
    last_monitor_boxes = []
    last_working_states = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

        # Run AI only every N frames
        # We lower the confidence threshold (conf=0.15) to detect laptops from side views better
        if frame_count % process_every_n_frames == 1:
            results = model(frame, verbose=False, conf=0.15)
            
            current_person_boxes = []
            current_monitor_boxes = []
            current_working_states = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if cls_name == 'person':
                        current_person_boxes.append([x1, y1, x2, y2])
                    elif cls_name in ['laptop', 'tvmonitor']:
                        current_monitor_boxes.append([x1, y1, x2, y2, cls_name])

            # Process each person
            for p_box in current_person_boxes:
                x1, y1, x2, y2 = p_box
                
                px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
                px2, py2 = min(width, x2 + 20), min(height, y2 + 20)
                person_roi = frame[py1:py2, px1:px2]
                
                is_working = False
                if person_roi.size > 0:
                    try:
                        landmarks = None
                        if pose is not None:
                            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                            pose_results = pose.process(roi_rgb)
                            landmarks = pose_results.pose_landmarks
                        
                        is_working = is_facing_and_sitting(landmarks, p_box, current_monitor_boxes)
                    except Exception as e:
                        pass
                
                current_working_states.append(is_working)
                
            # Update cache
            last_person_boxes = current_person_boxes
            last_monitor_boxes = current_monitor_boxes
            last_working_states = current_working_states

        # Draw monitor boxes
        for mb in last_monitor_boxes:
            x1, y1, x2, y2, cls_name = mb
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw person boxes and states
        for p_box, is_working in zip(last_person_boxes, last_working_states):
            x1, y1, x2, y2 = p_box
            color = (0, 255, 0) if is_working else (0, 0, 255)
            label = "Working" if is_working else "Not Working"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        
        # Display the frame
        display_frame = cv2.resize(frame, (800, int(800 * height / width)))
        cv2.imshow('Worker Activity Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing interrupted by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete! Output saved to '{args.output}'")

if __name__ == "__main__":
    main()
