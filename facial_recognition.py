import cv2
import os
import argparse
import urllib.request
import numpy as np

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {os.path.basename(filepath)}...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete.")

def load_models():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Modern High-Accuracy Face Detection (YuNet)
    detector_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    detector_model = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
    
    # Modern High-Accuracy Face Recognition (SFace)
    recognizer_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    recognizer_model = os.path.join(models_dir, "face_recognition_sface_2021dec.onnx")
    
    download_file(detector_url, detector_model)
    download_file(recognizer_url, recognizer_model)
    
    # Initialize the modern OpenCV face modules
    # Note: YuNet requires an input size, we'll set a default of 320x320 but we must update it per frame
    detector = cv2.FaceDetectorYN.create(detector_model, "", (320, 320), 0.6, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(recognizer_model, "")
    
    return detector, recognizer

def load_known_faces(known_dir, detector, recognizer):
    known_embeddings = {}
    if not os.path.exists(known_dir):
        os.makedirs(known_dir)
        print(f"Created directory {known_dir}. Please add some images of known people.")
        return known_embeddings

    print(f"Loading known faces from '{known_dir}'...")
    for filename in os.listdir(known_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(filename)[0]
            name = base_name.split('_')[0]
            
            img_path = os.path.join(known_dir, filename)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            (h, w) = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            
            if faces is not None and len(faces) > 0:
                # Get the first face (highest confidence)
                face = faces[0]
                
                # Align the face and extract features using SFace
                aligned_face = recognizer.alignCrop(img, face)
                feature = recognizer.feature(aligned_face)
                
                known_embeddings[name] = feature[0]
                print(f"Loaded {name}")
            else:
                print(f"Warning: No face found in {filename}")

    return known_embeddings

def cosine_similarity(feature1, feature2):
    return recognizer_match(feature1, feature2)

def recognizer_match(feature1, feature2):
    # SFace uses cosine similarity natively. We can compute it manually or use match()
    # cv2.FaceRecognizerSF expects shape (1, 128)
    f1 = feature1.flatten()
    f2 = feature2.flatten()
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))

def main():
    parser = argparse.ArgumentParser(description="Advanced Facial Recognition on Video using OpenCV SFace")
    parser.add_argument('--video', type=str, default='sample_traffic.mp4', help='Path to the input video file or camera index (e.g. 0)')
    parser.add_argument('--output', type=str, default='face_output.mp4', help='Path to save the output video')
    parser.add_argument('--known_dir', type=str, default='known_workers', help='Directory containing known faces')
    args = parser.parse_args()

    print("Loading High-Accuracy AI Models...")
    detector, recognizer = load_models()

    known_embeddings = load_known_faces(args.known_dir, detector, recognizer)
    if not known_embeddings:
        print("Warning: No known faces loaded. The system will only detect faces without recognizing them.")

    # Threshold for SFace cosine similarity is typically around 0.363 for strong matches
    similarity_threshold = 0.363

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

    print(f"Processing video '{args.video}'...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

        (h, w) = frame.shape[:2]
        detector.setInputSize((w, h))
        
        _, faces = detector.detect(frame)
        
        valid_faces = []
        if faces is not None:
            for face in faces:
                box = face[0:4].astype(int)
                x, y, fw, fh = box
                
                # Align and extract feature
                try:
                    aligned_face = recognizer.alignCrop(frame, face)
                    feature = recognizer.feature(aligned_face)
                    valid_faces.append({'box': (x, y, x+fw, y+fh), 'emb': feature[0], 'face_data': face})
                except Exception as e:
                    pass

        # Match faces to known embeddings uniquely
        matches = []
        for f_idx, face in enumerate(valid_faces):
            for known_key, known_emb in known_embeddings.items():
                score = recognizer_match(face['emb'], known_emb)
                if score > similarity_threshold:
                    matches.append((score, f_idx, known_key))

        matches.sort(key=lambda x: x[0], reverse=True)

        assigned_faces = {}
        assigned_names = set()

        for score, f_idx, known_key in matches:
            if f_idx not in assigned_faces and known_key not in assigned_names:
                assigned_faces[f_idx] = (known_key, score)
                assigned_names.add(known_key)

        # Draw boxes and labels
        for f_idx, face in enumerate(valid_faces):
            (startX, startY, endX, endY) = face['box']
            
            if f_idx in assigned_faces:
                best_match, best_score = assigned_faces[f_idx]
                color = (0, 255, 0)
                label = f"{best_match} ({best_score:.2f})"
            else:
                color = (0, 0, 255)
                label = "Unknown"
                
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, max(startY - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

        display_frame = frame
        if width > 800:
            display_frame = cv2.resize(frame, (800, int(800 * height / width)))
        cv2.imshow('High-Accuracy Facial Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete! Output saved to '{args.output}'")

if __name__ == "__main__":
    main()
