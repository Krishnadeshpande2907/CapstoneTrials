from ultralytics import YOLO
# import torch

# # Load the YOLO model (automatically detects GPU)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = YOLO('yolov10m').to(device)

# model.train(data=r"D:\VIT\4th year\8th Sem\Capstone\CapstoneTrials\archive\data.yaml", epochs=3, batch=4, imgsz=640, device="cuda")

from ultralytics import YOLO

# Load the model
model = YOLO("yolov10m.pt")

# Train the model
model.train(
    data="D:/VIT/4th year/8th Sem/Capstone/CapstoneTrials/archive/data.yaml",  # Path to your dataset YAML
    epochs=3,  # Increase epochs for better performance
    # batch=8,  # Reduce batch size for GPU memory constraints
    imgsz=640,
    device="cuda",  # Ensure training on GPU
    # optimizer="auto",  # Auto selects the best optimizer
    # workers=4,  # Reduce workers to optimize system performance
    # amp=False,  # Disable AMP to prevent NaN losses on GTX 1650 Ti
    # pretrained=True,  # Use pretrained weights
    # save=True,  # Save the model
    # val=True,  # Validate after training
    # patience=20,  # Allow patience for early stopping
    # cos_lr=True,  # Use cosine learning rate schedule
    # close_mosaic=10,  # Disable mosaic augmentation after 10 epochs
    # iou=0.7,  # Set IOU threshold
    # max_det=300,  # Maximum detections per image
    # save_period=5,  # Save every 5 epochs
    # freeze=None,  # Unfreeze all layers for better training
)

# Evaluate the model
metrics = model.val()

# Export the trained model
model.export(format="torchscript")  # Export in TorchScript format
