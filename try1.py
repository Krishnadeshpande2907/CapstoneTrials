import cv2
import torch
from ultralytics import YOLO

# Ensure model runs on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model (pre-trained) and move to GPU
model = YOLO("yolov8n.pt").to(device)

# Open a video stream (0 for webcam, or provide a video file path)
cap = cv2.VideoCapture(r"C:\Users\Krishna\Desktop\match.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on GPU
    results = model(frame, device=device)[0]  # Ensure model runs on CUDA

    # Loop through detections and draw bounding boxes
    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) == 0:  # Class 0 = 'person' in COCO dataset
            # Draw bounding box around detected person
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
