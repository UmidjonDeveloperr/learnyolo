import cv2
from ultralytics import YOLO
import json
import os
from datetime import datetime, timedelta

FPS = 25
INTERVAL_SECONDS = 60
OUTPUT_DIR = "/path-to-output"
SCREENSHOT_DIR = os.path.join(OUTPUT_DIR, "screenshots")

model = YOLO("/path-to-model/head_yolo11m_best.pt")
    

os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def parse_filename(filename: str):
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    room_part = parts[0]    
    start_time_str = parts[2]

    room_number = room_part.split("-")[0]
    start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M%S")

    return room_number, start_time


def count_people(frame,conf_threshold=0.5):
    results = model(frame, verbose=False)
    detections = results[0]

    if detections.boxes is None:
        return 0

    boxes = detections.boxes
    confidences = boxes.conf.cpu().numpy()
    people_count = int((confidences >= conf_threshold).sum())

    return people_count


def process_video(video_path):
    filename = os.path.basename(video_path)
    room_number, start_time = parse_filename(filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    json_path = os.path.join(OUTPUT_DIR, f"room_{room_number}.json")

    data = {
        "room": room_number,
        "video": filename,
        "start_time": start_time.isoformat(),
        "fps": FPS,
        "interval_seconds": INTERVAL_SECONDS,
        "records": []
    }

    interval_frames = FPS * INTERVAL_SECONDS
    frame_index = 0
    snapshot_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % interval_frames == 0:
            elapsed_seconds = frame_index / FPS
            real_time = start_time + timedelta(seconds=elapsed_seconds)

            people_count = count_people(frame)

            screenshot_name = f"room_{room_number}_{real_time.strftime('%H%M%S')}.jpg"
            screenshot_path = os.path.join(SCREENSHOT_DIR, screenshot_name)
            cv2.imwrite(screenshot_path, frame)

            record = {
                "index": snapshot_index,
                "timestamp": real_time.isoformat(),
                "elapsed_seconds": elapsed_seconds,
                "people_count": people_count,
                "screenshot": screenshot_name
            }

            data["records"].append(record)
            snapshot_index += 1

        frame_index += 1

    cap.release()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Processed room {room_number}: {len(data['records'])} records")


if __name__ == "__main__":
    video_file = "/path-to-video.mp4"
    process_video(video_file)
