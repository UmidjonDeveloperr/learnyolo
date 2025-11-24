from ultralytics import YOLO
import cv2
import cvzone

model = YOLO("path-to-head-detection-model/head_yolo11m_best.pt")
source_video = "/path-to-source-video/ClearPixCameraGroceryStoreFrontDoor.mp4"

cap = cv2.VideoCapture(source_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("result_tracking_ult_bs.mp4", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

line = [525, 297, 925, 297]
line1 = [525, 70, 925, 70]

entering_prev = {}
exiting_prev = {}
entering_people = []
exiting_people = []

resultsTracker = model.track(source=source_video, tracker="botsort.yaml", conf=0.4, iou=0.5, save=False, stream=True)

def is_enterance(cx, cy):
     if line[0] < cx < line[2] and line[1] > cy > line1[1]:
          return True
     return False

for result in resultsTracker:
    frame = result.orig_img
    boxes = result.boxes
    if boxes is None:
        continue
        
    for box in boxes:
        if box is None:
            continue

        x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
        if box.id is None:
            continue
        id = box.id.item()
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
                    
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (line1[0], line1[1]), (line1[2], line1[3]), (0, 0, 255), 5)
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 5)
            
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
        
    cv2.putText(frame, f"Entering people: {len(entering_people)}", (45, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Exiting people: {len(exiting_people)}", (45, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("BoT-SORT Implementation", frame)
    # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

            
