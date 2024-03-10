from ultralytics import YOLO

model = YOLO("YOLO_models/YOLOv8n-pose.pt")

results = model(source="data/man_walking.png", show=True, conf=0.3, save=True)