from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.info()
results = model.train(data="config.yaml", epochs=100, imgsz=640)
