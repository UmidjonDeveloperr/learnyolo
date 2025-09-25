from PIL import Image
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

source = "videos/cars_video.mp4"

result = model(source=source, show=True, conf=0.4, save=False)

for i, r in enumerate(result):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()


