import cv2
import os
import glob
import numpy as np

# Set the path to your frames directory
frames_path = r"D:\VIT\4th year\8th Sem\Capstone\Football\SoccerNet\SN-GSR-2025\train\SNGS-060\img1"  # Change this to your folder path

# Get a sorted list of frame filenames
frames = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))

# Check if frames exist
if not frames:
    print("No frames found in the specified directory!")
    exit()

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set frame delay (adjust based on FPS)
frame_delay = 50  # 30ms delay ~ 33 FPS

# Display frames
for frame in frames:
    img = cv2.imread(frame)
    if img is None:
        continue  # Skip if image is not loaded

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Video Playback", img)

    cv2.imshow("Video Playback", img)

    # Wait and break if 'q' is pressed
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()