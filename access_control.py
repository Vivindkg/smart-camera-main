import cv2
import os
import numpy as np

from facial_recognition import load_models, load_known_faces, recognizer_match

def draw_banner(frame, text, color):
    (h, w) = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 80), color, -1)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.5
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    text_x = (w - text_width) // 2
    text_y = 50
    
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

def main():
    print("Loading High-Accuracy Access Control System...")
    detector, recognizer = load_models()
    
    known_dir = 'known_workers'
    known_embeddings = load_known_faces(known_dir, detector, recognizer)
    
    if not known_embeddings:
        print("Warning: No known faces enrolled. Everyone will be DENIED.")

    similarity_threshold = 0.363 # SFace strict threshold

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("System Ready. Waiting for a face...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        detector.setInputSize((w, h))
        
        _, faces = detector.detect(frame)

        largest_face_area = 0
        largest_face_data = None
        largest_box = None

        if faces is not None:
            for face in faces:
                box = face[0:4].astype(int)
                x, y, fw, fh = box
                area = fw * fh
                
                # SFace confidence is face[-1]. For YuNet, we can just rely on the detection if it exists
                if area > largest_face_area and fw > 40 and fh > 40:
                    largest_face_area = area
                    largest_face_data = face
                    largest_box = box

        status_text = "SCANNING..."
        status_color = (150, 150, 150)
        
        if largest_face_data is not None:
            x, y, fw, fh = largest_box
            
            try:
                aligned_face = recognizer.alignCrop(frame, largest_face_data)
                feature = recognizer.feature(aligned_face)
                
                best_match = "Unknown"
                best_score = -1.0
                
                for known_key, known_emb in known_embeddings.items():
                    score = recognizer_match(feature[0], known_emb)
                    if score > best_score:
                        best_score = score
                        if score > similarity_threshold:
                            best_match = known_key.split('_')[0]
                            
                if best_match != "Unknown":
                    status_text = f"ACCESS GRANTED: {best_match.upper()}"
                    status_color = (0, 200, 0)
                    box_color = (0, 255, 0)
                else:
                    status_text = "ACCESS DENIED"
                    status_color = (0, 0, 200)
                    box_color = (0, 0, 255)
                    
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), box_color, 3)
                
            except Exception as e:
                pass
                
        draw_banner(frame, status_text, status_color)

        display_frame = frame
        if w > 1000:
            display_frame = cv2.resize(frame, (1000, int(1000 * h / w)))
            
        try:
            cv2.imshow('Smart Access Control', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error:
            pass

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

if __name__ == "__main__":
    main()
