# collect_data.py
from ultralytics import YOLO
import cv2
import time
import csv
import numpy as np

VIDEO_PATH = "traffic_video.mp4"   # or 0 for webcam
MODEL_PATH = "yolov8n.pt"
OUT_CSV = "traffic_counts.csv"

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# Simple zone split: top half = NS, bottom half = EW
def count_vehicles_zones(boxes, frame_h):
    ns, ew = 0, 0
    for b in boxes:
        if int(b.cls[0]) not in VEHICLE_CLASSES:
            continue
        y1, y2 = float(b.xyxy[0][1]), float(b.xyxy[0][3])
        y_center = (y1 + y2) / 2.0
        if y_center < frame_h / 2:
            ns += 1
        else:
            ew += 1
    return ns, ew

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_PATH}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_epoch", "ns", "ew", "total"])

        last_log = 0.0
        interval = 1.0  # seconds

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            if now - last_log >= interval:
                last_log = now

                res = model(frame, verbose=False)[0]
                ns, ew = count_vehicles_zones(res.boxes, frame.shape[0])
                total = ns + ew

                w.writerow([now, ns, ew, total])
                print(f"t={now:.0f} NS={ns} EW={ew} total={total}")

            cv2.imshow("collect_data (ESC to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved: {OUT_CSV}")

if __name__ == "__main__":
    main()