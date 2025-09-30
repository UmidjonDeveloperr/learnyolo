from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model = YOLO("yolov8n-seg.pt")
model = YOLO("yolov8n-pose.pt")

test_video = "videos/cars_video.mp4"

result = model.track(source=0, show=True, conf=0.4, iou=0.5, save=False)

for r in result:
    print(r.masks)
