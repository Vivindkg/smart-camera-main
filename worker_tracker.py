import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import argparse
import math
import os
import urllib.request

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")

def load_gender_model():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    prototxt_url = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_net.caffemodel"
    
    prototxt_path = os.path.join(models_dir, "gender_deploy.prototxt")
    caffemodel_path = os.path.join(models_dir, "gender_net.caffemodel")
    
    try:
        download_file(prototxt_url, prototxt_path)
        download_file(caffemodel_url, caffemodel_path)
        
        gender_net = cv2.dnn.readNet(caffemodel_path, prototxt_path)
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        return gender_net, face_cascade
    except Exception as e:
        print(f"Error loading gender model: {e}")
        return None, None

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_facing_and_sitting(pose_landmarks, person_box, monitor_boxes, phone_boxes):
    """
    Heuristic function to determine if a worker is likely sitting and facing a monitor.
    Uses proximity to the detected monitor and bounding box aspect ratio for posture.
    """
    # If no monitors are detected, they aren't facing a computer
    if not monitor_boxes:
        return False
        
    person_width = person_box[2] - person_box[0]
    person_height = person_box[3] - person_box[1]
    
    # Check if the person is using a phone
    px1, py1, px2, py2 = person_box
    for ph_box in phone_boxes:
        phx1, phy1, phx2, phy2 = ph_box
        # Check if the phone bounding box intersects with the person bounding box at all
        if not (px2 < phx1 or px1 > phx2 or py2 < phy1 or py1 > phy2):
            return False # Using phone, so not working
            
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
    parser.add_argument('--video', type=str, default='0', help='Path to the input video file (.mp4) or camera index (e.g. 0)')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to save the output video')
    args = parser.parse_args()

    # Load Gender Model
    print("Loading Gender Model...")
    gender_net, face_cascade = load_gender_model()
    gender_list = ['Male', 'Female']
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    # Initialize YOLOv8
    print("Loading YOLOv8 model...")
    try:
        model = YOLO('yolov8s.pt') # Upgraded from 'n' to 's' model for better small object (phone) detection
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

    video_source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{args.video}'")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Fallback for webcams which often report 0 FPS
        fps = 30.0

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing video '{args.video}'...")
    frame_count = 0
    
    # Optimization variables
    process_every_n_frames = 5 # Run AI every 5th frame
    next_worker_id = 0
    tracked_workers = {} # id -> {'box': [x1,y1,x2,y2], 'is_working': False, 'working_frames': 0, 'not_working_frames': 0, 'gender': 'Unknown'}
    last_monitor_boxes = []
    last_phone_boxes = []
    
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
            current_phone_boxes = []
            current_working_states = []
            current_genders = []

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
                    elif cls_name == 'cell phone':
                        current_phone_boxes.append([x1, y1, x2, y2])

            # Process each person
            for p_box in current_person_boxes:
                x1, y1, x2, y2 = p_box
                
                px1, py1 = max(0, x1 - 20), max(0, y1 - 20)
                px2, py2 = min(width, x2 + 20), min(height, y2 + 20)
                person_roi = frame[py1:py2, px1:px2]
                
                is_working = False
                detected_gender = None
                
                if person_roi.size > 0:
                    try:
                        # Gender detection
                        if gender_net is not None and face_cascade is not None:
                            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            if len(faces) > 0:
                                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                                fx, fy, fw, fh = faces[0]
                                face_img = person_roi[fy:fy+fh, fx:fx+fw]
                                
                                if face_img.size > 0:
                                    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                                    gender_net.setInput(blob)
                                    gender_preds = gender_net.forward()
                                    detected_gender = gender_list[gender_preds[0].argmax()]
                                    
                        landmarks = None
                        if pose is not None:
                            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                            pose_results = pose.process(roi_rgb)
                            landmarks = pose_results.pose_landmarks
                        
                        is_working = is_facing_and_sitting(landmarks, p_box, current_monitor_boxes, current_phone_boxes)
                    except Exception as e:
                        pass
                
                current_working_states.append(is_working)
                current_genders.append(detected_gender)
                
            # Update trackers (associate new boxes with existing workers based on distance)
            new_tracked_workers = {}
            for p_box, is_working, p_gender in zip(current_person_boxes, current_working_states, current_genders):
                cx = (p_box[0] + p_box[2]) / 2
                cy = (p_box[1] + p_box[3]) / 2
                
                best_id = None
                min_dist = float('inf')
                for wid, data in tracked_workers.items():
                    old_cx = (data['box'][0] + data['box'][2]) / 2
                    old_cy = (data['box'][1] + data['box'][3]) / 2
                    dist = math.sqrt((cx-old_cx)**2 + (cy-old_cy)**2)
                    if dist < 200 and dist < min_dist: # Increased distance threshold slightly to help reconnect lost workers
                        min_dist = dist
                        best_id = wid
                        
                if best_id is not None:
                    worker_data = tracked_workers.pop(best_id)
                    worker_data['box'] = p_box
                    worker_data['lost_frames'] = 0 # Reset lost counter
                    
                    if p_gender is not None:
                        worker_data['gender'] = p_gender
                    
                    if not is_working:
                        worker_data['not_working_streak'] = worker_data.get('not_working_streak', 0) + 1
                    else:
                        worker_data['not_working_streak'] = 0
                        
                    # State transition with buffer
                    if worker_data['not_working_streak'] > 3:
                        worker_data['is_working'] = False
                    elif worker_data['not_working_streak'] > 0 and worker_data.get('is_working', False):
                        # Still in buffer, keep state as working
                        worker_data['is_working'] = True
                    else:
                        worker_data['is_working'] = is_working
                        
                    new_tracked_workers[best_id] = worker_data
                else:
                    new_tracked_workers[next_worker_id] = {
                        'box': p_box,
                        'is_working': is_working,
                        'working_frames': 0,
                        'not_working_frames': 0,
                        'not_working_streak': 0 if is_working else 1,
                        'lost_frames': 0,
                        'gender': p_gender if p_gender is not None else 'Unknown'
                    }
                    next_worker_id += 1
                    
            # Handle lost workers (those not matched to any current detection)
            for wid, worker_data in tracked_workers.items():
                worker_data['lost_frames'] = worker_data.get('lost_frames', 0) + 1
                if worker_data['lost_frames'] < 10: # Keep their identity alive for ~1.5 seconds if YOLO loses them
                    new_tracked_workers[wid] = worker_data
                    
            tracked_workers = new_tracked_workers
            last_monitor_boxes = current_monitor_boxes
            last_phone_boxes = current_phone_boxes

        # For every frame, increment the appropriate timer
        for wid, data in tracked_workers.items():
            # If the worker is temporarily lost by the AI, pause their timers
            if data.get('lost_frames', 0) > 0:
                continue
                
            if data['is_working']:
                data['working_frames'] = data.get('working_frames', 0) + 1
            else:
                data['not_working_frames'] = data.get('not_working_frames', 0) + 1

        # Draw monitor boxes
        for mb in last_monitor_boxes:
            x1, y1, x2, y2, cls_name = mb
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
        # Draw phone boxes
        for pb in last_phone_boxes:
            x1, y1, x2, y2 = pb
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, "phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Draw person boxes and timer
        for wid, data in tracked_workers.items():
            x1, y1, x2, y2 = data['box']
            is_working = data['is_working']
            working_time = data.get('working_frames', 0) / fps
            break_time = data.get('not_working_frames', 0) / fps
            
            color = (0, 255, 0) if is_working else (0, 0, 255)
            # Display cumulative times
            gender = data.get('gender', 'Unknown')
            if gender != 'Unknown':
                label = f"{gender} - Work: {working_time:.1f}s | Break: {break_time:.1f}s"
            else:
                label = f"Work: {working_time:.1f}s | Break: {break_time:.1f}s"
            
            # Add a small padding to the bounding box so it surrounds the person a bit wider
            px1, py1 = max(0, x1 - 15), max(0, y1 - 15)
            px2, py2 = min(width, x2 + 15), min(height, y2 + 15)
            
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2) # Reduced thickness from 3 to 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (px1, py1 - 25), (px1 + tw, py1), color, -1)
            cv2.putText(frame, label, (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        
        # Display the frame
        if width > 800:
            display_frame = cv2.resize(frame, (800, int(800 * height / width)))
        else:
            display_frame = frame
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
