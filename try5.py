import cv2
from ultralytics import YOLO
import torch

# Load the YOLO model (automatically detects GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO('yolov10x').to(device)

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

    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class label
                
                label = f"{model.names[cls]}: {conf:.2f}"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLO Object Detection", frame)
    # out.write(frame)  # Uncomment to save output video

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()  # Uncomment if saving video
cv2.destroyAllWindows()