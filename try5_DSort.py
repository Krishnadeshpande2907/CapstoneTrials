import cv2
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the YOLO model (automatically detects GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO('yolov10x').to(device)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)  # max_age defines how long an object is tracked without new detections

# Open the video file
video_path = r"d:\VIT\4th year\8th Sem\Capstone\CapstoneTrials\08fd33_4.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video file is loaded
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# VideoWriter to save output (optional)
# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error reading frame.")
        break  # Exit loop if no frame is read

    # Run YOLO model on the frame
    results = model.predict(frame, device=device)

    detections = []  # List to store detected objects for tracking
    
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class label
                
                if model.names[cls] == "person":  # Track only players
                    detections.append(([x1, y1, x2, y2], conf, cls))
    
    # Update tracker with new detections
    tracks = tracker.update_tracks(detections, frame=frame)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get bounding box in left-top-right-bottom format
        x1, y1, x2, y2 = map(int, ltrb)
        
        label = f"ID {track_id}"

        # Draw smaller bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Show the frame
    cv2.imshow("YOLO + DeepSORT Tracking", frame)
    # out.write(frame)  # Uncomment to save output video

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()  # Uncomment if saving video
cv2.destroyAllWindows()