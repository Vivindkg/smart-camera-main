import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Enroll a new face into the known_workers directory")
    parser.add_argument('--name', type=str, required=True, help="The name of the person you are enrolling (e.g., 'Vikas')")
    parser.add_argument('--video', type=str, default='0', help="Path to a video file, or '0' for webcam")
    args = parser.parse_args()

    known_dir = "known_workers"
    os.makedirs(known_dir, exist_ok=True)

    # Clean the name to be a valid filename
    safe_name = "".join([c for c in args.name if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    filepath = os.path.join(known_dir, f"{safe_name}.jpg")

    video_source = int(args.video) if args.video.isdigit() else args.video
    
    if args.video == '0':
        print(f"Opening webcam to capture face for: {safe_name}")
    else:
        print(f"Opening video '{args.video}' to capture face for: {safe_name}")
        
    print("Press 'c' to capture the photo, or 'q' to quit.")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load a quick Haar Cascade just to help the user center their face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(display_frame, "Center face here and press 'c'", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Enroll New Face', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save the ORIGINAL frame (without the blue box drawn on it)
            cv2.imwrite(filepath, frame)
            print(f"Success! Face saved to {filepath}")
            print(f"The system will now recognize {safe_name} automatically next time you run facial_recognition.py")
            break
        elif key == ord('q'):
            print("Enrollment cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
