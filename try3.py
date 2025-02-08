import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Ensure model runs on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model (pre-trained) and move to GPU
model = YOLO("yolov8n.pt").to(device)

# Open a video stream (0 for webcam, or provide a video file path)
cap = cv2.VideoCapture(r"C:\Users\Krishna\Desktop\match.mp4")

# Define Kalman Filter class
class KalmanTracker:
    def __init__(self, bbox):
        """
        Initialize Kalman filter with the initial bounding box.
        """
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.lost = 0  # Track if detection is lost
        
        # State: [x, y, w, h, dx, dy]
        self.state = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], 0, 0], dtype=np.float32)
        
        # State transition matrix
        self.F = np.eye(6)
        self.F[0, 4] = 1  # x += dx
        self.F[1, 5] = 1  # y += dy
        
        # Measurement matrix
        self.H = np.eye(4, 6)
        
        # Process noise
        self.Q = np.eye(6) * 0.01
        
        # Measurement noise
        self.R = np.eye(4) * 0.1
        
        # Covariance
        self.P = np.eye(6)

    def predict(self):
        """
        Predict the next state using the Kalman filter.
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, bbox):
        """
        Update the state using the new detection.
        """
        z = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float32)
        
        # Compute Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = z - self.H @ self.state
        self.state += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_bbox(self):
        """
        Return bounding box.
        """
        x, y, w, h = self.state[:4]
        return [int(x), int(y), int(x + w), int(y + h)]

# Initialize tracker list
trackers = []
KalmanTracker.count = 0
MAX_IDS = 25  # Limit to 22 players + 3 referees
DETECTION_TIMEOUT = 20  # Number of frames without detection before ending trajectory

def iou(bb1, bb2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    """
    x1, y1, x2, y2 = bb1
    x1g, y1g, x2g, y2g = bb2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def reidentify_trackers(trackers, detections):
    """
    Re-identify trackers based on IoU and position.
    """
    unassigned_trackers = []
    assigned_detections = []

    for tracker in trackers:
        best_match = None
        best_iou = 0
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                iou_score = iou(tracker.get_bbox(), det)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_match = i
        if best_match is not None:
            tracker.update(detections[best_match])
            assigned_detections.append(best_match)
            tracker.lost = 0  # Reset lost counter when detection is matched
        else:
            tracker.lost += 1  # Increment lost counter when detection is not found
            if tracker.lost > DETECTION_TIMEOUT:
                unassigned_trackers.append(tracker)  # Mark tracker for removal if it lost too many times

    return assigned_detections, unassigned_trackers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on GPU
    results = model(frame, device=device)[0]  # Ensure model runs on CUDA
    detections = []

    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) == 0:  # Class 0 = 'person'
            detections.append([int(x1), int(y1), int(x2), int(y2)])

    # Predict new locations for existing trackers
    for tracker in trackers:
        tracker.predict()

    # Re-identify and assign trackers
    assigned_detections, unassigned_trackers = reidentify_trackers(trackers, detections)

    # Remove trackers that were lost for too long
    trackers = [tracker for tracker in trackers if tracker not in unassigned_trackers]

    # Create new trackers for unmatched detections if within the limit
    for j, det in enumerate(detections):
        if j not in assigned_detections and len(trackers) < MAX_IDS:
            trackers.append(KalmanTracker(det))

    # Draw bounding boxes and IDs
    for tracker in trackers:
        x1, y1, x2, y2 = tracker.get_bbox()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {tracker.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Object Detection & Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
