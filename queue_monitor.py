import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import time
import winsound

def is_inside_polygon(point, polygon):
    # cv2.pointPolygonTest returns positive if inside, negative if outside, 0 if on the edge
    return cv2.pointPolygonTest(polygon, point, False) >= 0

drawing_points = []
def draw_polygon(event, x, y, flags, param):
    global drawing_points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))

def get_interactive_roi(frame):
    global drawing_points
    drawing_points = []
    clone = frame.copy()
    window_name = "Draw Queue Box (Click points, press ENTER to finish)"
    
    # Resize for drawing if it's too big
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 1200:
        scale = 1200 / w
        clone = cv2.resize(clone, (1200, int(h * scale)))
        
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon)
    
    print("Please draw the queue box by clicking on the image.")
    print("Press ENTER when you are done.")
    
    while True:
        display_frame = clone.copy()
        for pt in drawing_points:
            cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
        if len(drawing_points) > 1:
            cv2.polylines(display_frame, [np.array(drawing_points)], isClosed=False, color=(0, 255, 255), thickness=2)
        if len(drawing_points) > 2:
            cv2.polylines(display_frame, [np.array(drawing_points)], isClosed=True, color=(0, 255, 255), thickness=1)
            
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == ord('q'): # Enter or q
            break
            
    cv2.destroyWindow(window_name)
    
    if len(drawing_points) < 3:
        print("Not enough points drawn. Using default box.")
        return np.array([
            [int(w * 0.25), int(h * 0.2)],
            [int(w * 0.75), int(h * 0.2)],
            [int(w * 0.85), int(h * 0.9)],
            [int(w * 0.15), int(h * 0.9)]
        ], np.int32)
        
    # Scale points back to original resolution
    return np.array([(int(pt[0]/scale), int(pt[1]/scale)) for pt in drawing_points], np.int32)


def main():
    parser = argparse.ArgumentParser(description="Queue Monitor: Count people standing in a specific line/area")
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index')
    parser.add_argument('--output', type=str, default='queue_output.mp4', help='Path to save output video')
    args = parser.parse_args()

    print("Loading YOLOv8 model for precise people counting...")
    # Using YOLOv8 small model
    model = YOLO('yolov8s.pt') 
    
    # Class 0 in COCO dataset is 'person'
    PERSON_CLASS = 0

    is_image = args.video.lower().endswith(('.png', '.jpg', '.jpeg'))
    
    if is_image:
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

    if not is_image:
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading video stream")
            return
    else:
        first_frame = frame.copy()

    # Define the Region of Interest (ROI) interactively
    roi_polygon = get_interactive_roi(first_frame)

    # If it's a video, process the first frame we grabbed for drawing
    if not is_image:
        frame = first_frame
        
    print(f"Processing '{args.video}'...")
    
    last_alarm_time = 0
    
    while True:
        loop_start = time.time()
        
        # Run YOLOv8 detection
        results = model(frame, classes=[PERSON_CLASS], verbose=False)
        
        # Draw the ROI polygon on the frame
        cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 255, 0), thickness=2)
        
        # Add a slightly transparent overlay for the ROI to make it obvious
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_polygon], (255, 255, 0))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "QUEUE ZONE", (int(width * 0.3), int(height * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        people_in_line = 0
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu()
            confidences = results[0].boxes.conf.cpu()
            
            for box, conf in zip(boxes, confidences):
                if conf < 0.4:  # Minimum confidence threshold
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                
                # Use the bottom center of the bounding box (person's feet) to determine if they are in the zone
                feet_x = int((x1 + x2) / 2)
                feet_y = y2
                
                in_line = is_inside_polygon((feet_x, feet_y), roi_polygon)
                
                if in_line:
                    people_in_line += 1
                    color = (0, 255, 0)  # Green for in line
                    label = "In Line"
                else:
                    color = (0, 0, 255)  # Red for out of line
                    label = "Ignoring"
                    
                # Draw Box and Info
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (feet_x, feet_y), 5, color, -1)  # Draw point at feet
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw overall count banner
        cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 0), -1)
        count_color = (0, 255, 0) if people_in_line <= 7 else (0, 0, 255)
        cv2.putText(frame, f"People in Line: {people_in_line}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, count_color, 3)
        
        if people_in_line > 7:
            cv2.putText(frame, "ALARM: QUEUE TOO LONG!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if time.time() - last_alarm_time > 2.0:
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                last_alarm_time = time.time()
        
        # Display
        display_frame = frame
        if width > 1000:
            display_frame = cv2.resize(frame, (1000, int(1000 * height / width)))
            
        cv2.imshow('Precise Queue Monitor', display_frame)

        if is_image:
            cv2.imwrite(args.output.replace('.mp4', '.jpg'), frame)
            print("Press any key in the image window to close it...")
            cv2.waitKey(0)
            break
        else:
            out.write(frame)
            
            loop_time = time.time() - loop_start
            wait_ms = max(1, int(1000 / fps) - int(loop_time * 1000))
            
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
                
        # Grab next frame if video
        if not is_image:
            ret, frame = cap.read()
            if not ret:
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
