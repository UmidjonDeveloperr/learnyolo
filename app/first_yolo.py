from ultralytics import YOLO

model = YOLO("yolov8n.pt")

source = "videos/cars_video.mp4"

result = model.train(data="coco8.yaml", epochs=100, imgsz=640)

result = model.predict(source=0, show=True, conf=0.4, save=False)


