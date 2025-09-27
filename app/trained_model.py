from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

result = model.predict(source=0, show=True, conf=0.4, save=False)

metrics = model.val()
print(metrics.box.map)