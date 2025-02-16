import json
import cv2
import os

# Load the sequences info
try:
    with open('D:/VIT/4th year/8th Sem/Capstone/Football/SoccerNet/SN-GSR-2025/train/sequences_info.json', 'r') as f:
        data = json.load(f)
    print("Loaded sequences info successfully.")
except Exception as e:
    print(f"Failed to load sequences info: {e}")

# json_file_path = '../Football/SoccerNet/SN-GSR-2025/train/sequences_info.json'

# Load the JSON file
# with open(json_file_path, 'r') as file:
#     data = json.load(file)

# Extract the sequence of image filenames
# Assuming the images are named sequentially based on the 'name' field in the JSON
# image_sequence = [f"{item['name']}.jpg" for item in data['train']]

# file sequence
file_sequence = [f"{item['name']}" for item in data['train']]

# Directory containing the image files
# image_directory = '/path/to/your/image_directory/'  # Update this path to your actual image directory

# file directory
file_directory = 'D:/VIT/4th year/8th Sem/Capstone/Football/SoccerNet/SN-GSR-2025/train/'

'''
# Display the images in sequence
for image_filename in image_sequence:
    image_path = os.path.join(image_directory, image_filename)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error loading image: {image_path}")
        continue
    
    cv2.imshow('Video', image)
    
    # Wait for 30 milliseconds before displaying the next image
    # Press 'q' to exit the video early
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
'''

# Display the images in sequence
for file_sequence in file_sequence:
    # inside the file sequence go into folder named img1. there are the photos
    image_path = os.path.join(file_directory, file_sequence, 'img1')
    # image_path = os.path.join(file_directory, image_filename)
    # Check if the directory exists
    if not os.path.exists(image_path):
        print(f"Directory does not exist: {image_path}")
        continue

    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')])
    
    for image_filename in image_files:
        image_path = os.path.join(image_path, image_filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        
        cv2.imshow('Video', image)
        
        # Wait for 30 milliseconds before displaying the next image
        # Press 'q' to exit the video early
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# Release the OpenCV window
cv2.destroyAllWindows()