from collections import defaultdict
import cv2

def process_video_for_counting_with_tracking(video_path, model):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- FIX 3: Restrict line width ---
    # Change these values if you only want to count people crossing a specific part of the screen
    line_start_x = 0  # e.g., change to frame_width // 4 to cut off the left side
    line_end_x = frame_width # e.g., change to (frame_width // 4) * 3 to cut off the right side

    counting_line_y = frame_height // 3 
    
    # --- FIX 1: Increase buffer zone ---
    # Increased from 20 to 40 to better handle bounding box jitter
    buffer_pixels = 40 
    upper_line_y = counting_line_y - buffer_pixels
    lower_line_y = counting_line_y + buffer_pixels

    tracker = Tracker() # Assuming your tracker class

    # We are adding a new state: 'pending_in_buffer'
    object_state = {}

    entry_count = 0
    exit_count = 0

    print("Starting video processing with tracking...")

    output_video_path = 'output_counted_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=False, verbose=False)

        current_detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if model.names[int(box.cls[0])] == 'person':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Optional: Use the bottom-center (feet) instead of true center for more stable tracking
                    # centroid_x = (x1 + x2) // 2
                    # centroid_y = y2 
                    
                    current_detections.append((x1, y1, x2, y2))

        objects = tracker.update(current_detections)

        current_tracker_ids = set(objects.keys())
        for obj_id in list(object_state.keys()):
            if obj_id not in current_tracker_ids:
                del object_state[obj_id]

        for object_id, centroid in objects.items():
            centroid_x, centroid_y = centroid

            # Draw centroid
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 255), -1)

            # Determine actual zone
            actual_centroid_zone = None
            actual_centroid_zone_text = "Unknown"

            if centroid_y < upper_line_y:
                actual_centroid_zone = 'above'
                actual_centroid_zone_text = 'Above'
            elif centroid_y > lower_line_y:
                actual_centroid_zone = 'below'
                actual_centroid_zone_text = 'Below'
            else:
                actual_centroid_zone = 'in_buffer'
                actual_centroid_zone_text = 'Buffer'

            # Ensure the person is within our defined X-boundaries before tracking their state
            # If they are outside the horizontal line segment, we ignore them
            if not (line_start_x <= centroid_x <= line_end_x):
                 # Display they are out of bounds and skip state logic
                 cv2.putText(frame, f"ID:{object_id} (Out of Bounds)", (centroid_x - 10, centroid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                 continue 

            # --- FIX 2: Better State Machine Initialization ---
            if object_id not in object_state: 
                if actual_centroid_zone == 'above':
                    object_state[object_id] = 'above_ready_to_enter'
                elif actual_centroid_zone == 'below':
                    object_state[object_id] = 'below_ready_to_exit'
                else: 
                    # If first seen inside the buffer, do not assume a state!
                    # Wait for them to fully exit the buffer to become ready.
                    object_state[object_id] = 'pending_in_buffer'
            
            current_obj_state = object_state[object_id]

            # Apply state transitions and count
            if current_obj_state == 'pending_in_buffer':
                # Wait until they definitively leave the buffer to assign a real state
                if actual_centroid_zone == 'above':
                    object_state[object_id] = 'above_ready_to_enter'
                elif actual_centroid_zone == 'below':
                    object_state[object_id] = 'below_ready_to_exit'

            elif current_obj_state == 'above_ready_to_enter':
                if actual_centroid_zone == 'below':
                    entry_count += 1
                    object_state[object_id] = 'crossing_down'

            elif current_obj_state == 'crossing_down':
                if actual_centroid_zone == 'above': 
                    object_state[object_id] = 'below_ready_to_exit' 

            elif current_obj_state == 'below_ready_to_exit':
                if actual_centroid_zone == 'above':
                    exit_count += 1
                    object_state[object_id] = 'crossing_up'

            elif current_obj_state == 'crossing_up':
                if actual_centroid_zone == 'below': 
                    object_state[object_id] = 'above_ready_to_enter' 

            # Display ID and state
            cv2.putText(frame, f"ID:{object_id} ({actual_centroid_zone_text})", (centroid_x - 10, centroid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw counting lines (Restricted to line_start_x and line_end_x)
        cv2.line(frame, (line_start_x, counting_line_y), (line_end_x, counting_line_y), (255, 0, 0), 2) 
        cv2.line(frame, (line_start_x, upper_line_y), (line_end_x, upper_line_y), (0, 255, 255), 1) 
        cv2.line(frame, (line_start_x, lower_line_y), (line_end_x, lower_line_y), (0, 255, 255), 1) 
        
        cv2.putText(frame, f"Buffer Zone ({buffer_pixels}px)", (line_start_x + 5, counting_line_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Display counts
        cv2.putText(frame, f"Entries: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Exits: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release() 
    cv2.destroyAllWindows()
    print(f"Processing finished. Total Entries: {entry_count}, Total Exits: {exit_count}")
