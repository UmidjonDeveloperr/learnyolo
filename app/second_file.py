import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (w, h)
output = cv2.VideoWriter("videos/camera_detected.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, frame_size)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        detected_frame = results[0].plot()
        cv2.imshow("frame", detected_frame)
        # output.write(detected_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()