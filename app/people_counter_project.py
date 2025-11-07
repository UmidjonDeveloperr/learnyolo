from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from SORT import *
import math
import cvzone

model = YOLO("/home/umidjon/cv_projects/learnyolo/app/models/head_detection/head_yolo11m_best.pt")

source_video = "/home/umidjon/cv_projects/learnyolo/app/videos/ClearPixCameraGroceryStoreFrontDoor.mp4"

cap = cv2.VideoCapture(source_video)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('tracking_output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

line = [525, 297, 925, 297]
line1 = [525, 70, 925, 70]

# line = [525, 370, 925, 370]
# line1 = [525, 150, 925, 150]

mask = cv2.imread("/home/umidjon/cv_projects/learnyolo/app/images/design_bitwise.png")
mask = cv2.resize(mask, (width, height))

total_people = []
prev_passed = {}
entering_prev = {}
exiting_prev = {}
entering_people = []
exiting_people = []

def is_enterance(cx, cy):
     if line[0] < cx < line[2] and line[1] > cy > line1[1]:
          return True
     return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # imgRegion = cv2.bitwise_and(frame, mask)
    results = model(frame, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[[0]] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.3 and cls == 0:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
        scale=2, thickness=3, offset=10)
                
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (line1[0], line1[1]), (line1[2], line1[3]), (0, 0, 255), 5)
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 5)
        
        cv2.putText(frame, f"Entering people: {len(entering_people)}", (45, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Exiting people: {len(exiting_people)}", (45, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        
        if is_enterance(cx, cy):
            if id in entering_prev:
                if len(entering_prev[id]) < 3 and entering_prev[id][-1] < cy:
                    entering_prev[id].append(cy)
                if len(entering_prev[id]) == 3:
                    if entering_people.count(id) == 0:
                        entering_people.append(id)

            if id in exiting_prev:
                if len(exiting_prev[id]) < 3 and exiting_prev[id][-1] > cy:
                    exiting_prev[id].append(cy)
                if len(exiting_prev[id]) == 3:
                    if exiting_people.count(id) == 0:
                        exiting_people.append(id)

            if (id not in entering_prev) and (id not in exiting_prev):
                entering_prev[id] = [cy]
                exiting_prev[id] = [cy]
        
    cv2.imshow("Image", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
out.release()
cv2.destroyAllWindows()

            
