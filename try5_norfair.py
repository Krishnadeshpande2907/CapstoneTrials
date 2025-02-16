# norfair is a substitute to ByteTrack, which is a library for object tracking

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from norfair import Tracker, Detection, draw_tracked_objects

# Load YOLOv10 model and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov10x").to(device)

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

# Initialize Norfair Tracker (Euclidean distance-based tracking)
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# VideoWriter to save output (optional)
# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error reading frame.")
        break  # Exit loop if no frame is read

    # Run YOLO model on the frame
    results = model.predict(frame, device=device)

    detections = []
    
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                conf = box.conf[0].item()  # Confidence score