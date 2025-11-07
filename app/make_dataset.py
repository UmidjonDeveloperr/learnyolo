import os
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

source = "videos/217-xona_253.mp4"

cap = cv2.VideoCapture(source)

frame_counter = 0
image_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1

    if frame_counter == 40:
        results = model(frame, conf=0.3)

        H, W =frame.shape[:2]

        image_counter += 1
        cv2.imwrite(f"/content/drive/MyDrive/detections_253/head_detected_images_253/217_253_img{image_counter}.jpg", frame)
        label_path = os.path.join("/content/drive/MyDrive/detections_253/head_detected_labels_253", f"217_253_img{image_counter}.txt")

        with open(label_path, "w") as f:
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    x_center = ((x1 + x2) / 2) / W
                    y_center = ((y1 + y2) / 2) / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H

                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:6f}\n")
        frame_counter = 0
        print(f"Saved image and labels {image_counter}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Saved image and labels, total count is {image_counter}")

cap.release()
cv2.destroyAllWindows()
