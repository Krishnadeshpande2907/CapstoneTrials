import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO('yolov10x').to(device)

# Initialize Deep OC-SORT tracker
tracker = DeepSort(max_age=30)  # Increase max_age if tracking is lost too soon

# Open the video file
video_path = r"d:\VIT\4th year\8th Sem\Capstone\CapstoneTrials\08fd33_4.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file is loaded
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error reading frame.")
        break  # Exit loop if no frame is read

    # Run YOLO model to detect objects
    results = model.predict(frame, device=device)
    
    detections = []
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class label

                if model.names[cls] == "person" and conf > 0.5:  # Track only players
                    detections.append([[x1, y1, x2, y2], conf, cls])

    # Update tracker with new detections
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and tracking IDs
    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)  # Bounding box coordinates

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Deep OC-SORT Player Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()