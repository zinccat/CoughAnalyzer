from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

# Validate with IoU=0.4 for NMS
metrics = model.val(data="data.yaml", conf=0.3, iou=0.4, plots=True)
