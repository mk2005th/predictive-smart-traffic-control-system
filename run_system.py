from ultralytics import YOLO
import cv2
import time
import numpy as np
from collections import deque
from esn import EchoStateNetwork

VIDEO_PATH = "traffic_video.mp4"
# VIDEO_PATH = 0 
MODEL_PATH = "yolov8n.pt"
ESN_PATH   = "esn_model.npz"
NORM_PATH  = "norm_params.npz"

# display size (fits most laptops)
DISP_W, DISP_H = 960, 540

VEHICLE_CLASSES = {2, 3, 5, 7}

BASE_GREEN = 20.0
MAX_GREEN  = 60.0

SMOOTH_WINDOW = 7
TICK_INTERVAL = 1.0
PROACTIVE_THRESHOLD = 0.70

# graph history
HIST_N = 180  # last 3 mins if tick=1s

# ✅ NEW: if smoothed total is below this, force no congestion
LOW_TRAFFIC_THRESHOLD = 2.0  # tune: 1.0/2.0/3.0 based on your webcam scene


def count_vehicles_zones(boxes, frame_h):
    ns, ew = 0, 0
    for b in boxes:
        if int(b.cls[0]) not in VEHICLE_CLASSES:
            continue
        y1, y2 = float(b.xyxy[0][1]), float(b.xyxy[0][3])
        yc = (y1 + y2) / 2.0
        if yc < frame_h / 2:
            ns += 1
        else:
            ew += 1
    return ns, ew


def compute_green_time(count, other_count):
    total = max(1, count + other_count)
    share = count / total
    return float(np.clip(BASE_GREEN + share * (MAX_GREEN - BASE_GREEN), BASE_GREEN, MAX_GREEN))


def normalize_vec(x, mu, sd):
    return (x - mu) / sd


def draw_header(frame, ns, ew, green_dir, time_left, green_alloc, p_cong):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "Smart Traffic: Dynamic Control + ESN Prediction",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"NS={ns}  EW={ew}  GREEN={green_dir}  left={time_left:0.1f}s  alloc={green_alloc:0.0f}s",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    col = (0, 255, 0) if p_cong < 0.4 else (0, 165, 255) if p_cong < 0.7 else (0, 0, 255)
    cv2.putText(frame, f"Predicted congestion soon: p={p_cong*100:0.1f}%",
                (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)


def plot_series(panel, series, x0, y0, w, h, y_min, y_max, title, color):
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (35, 35, 35), -1)
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (120, 120, 120), 1)
    cv2.putText(panel, title, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if len(series) < 2:
        return

    s = np.array(series, dtype=np.float32)
    s = np.clip(s, y_min, y_max)

    def to_xy(i, val):
        x = x0 + int(i * (w - 20) / (len(s) - 1)) + 10
        yn = (val - y_min) / (y_max - y_min + 1e-6)
        y = y0 + h - 10 - int(yn * (h - 40))
        return x, y

    pts = [to_xy(i, s[i]) for i in range(len(s))]
    for i in range(len(pts) - 1):
        cv2.line(panel, pts[i], pts[i + 1], color, 2)

    cv2.circle(panel, pts[-1], 4, (255, 255, 255), -1)
    cv2.putText(panel, f"{float(series[-1]):.2f}", (x0 + w - 80, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    yolo = YOLO(MODEL_PATH)

    esn = EchoStateNetwork.load(ESN_PATH)
    norm = np.load(NORM_PATH)
    mu = norm["mu"].astype(np.float32).reshape(-1)
    sd = norm["sd"].astype(np.float32).reshape(-1)

    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_PATH}")

    total_hist = deque(maxlen=50)
    ma_hist = deque(maxlen=SMOOTH_WINDOW)

    green_dir = "NS"
    green_alloc = 30.0
    switch_at = time.time() + green_alloc

    last_tick = 0.0
    p_cong = 0.0

    series_total = deque(maxlen=HIST_N)
    series_p = deque(maxlen=HIST_N)

    # ✅ NEW: track last state of "low traffic" to reset ESN only when entering low-traffic
    was_low = False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)
        now = time.time()

        res = yolo(frame, verbose=False)[0]
        ns, ew = count_vehicles_zones(res.boxes, frame.shape[0])
        total = ns + ew

        ma_hist.append(total)
        total_smooth = float(np.mean(ma_hist))

        if now - last_tick >= TICK_INTERVAL:
            last_tick = now

            total_hist.append(total_smooth)
            rate = float(total_hist[-1] - total_hist[-2]) if len(total_hist) >= 2 else 0.0
            ma3 = float(np.mean(list(total_hist)[-3:])) if len(total_hist) >= 3 else total_smooth

            # ✅ LOW-TRAFFIC GUARD
            low = total_smooth < LOW_TRAFFIC_THRESHOLD

            if low:
                # Reset ESN once when we enter low-traffic mode (prevents "stuck at 0.97")
                if not was_low:
                    try:
                        esn.reset_state()
                    except Exception:
                        pass

                p_cong = 0.0  # congestion cannot be soon if there are ~0 vehicles
            else:
                feat = np.array([ns, ew, total_smooth, rate, ma3], dtype=np.float32)
                feat_n = normalize_vec(feat, mu, sd)
                p_cong = float(esn.predict_proba(feat_n)[0])

            was_low = low

            series_total.append(total_smooth)
            series_p.append(p_cong)

            # proactive: only if NOT low traffic
            if (not was_low) and (p_cong >= PROACTIVE_THRESHOLD):
                green_dir = "NS" if ns >= ew else "EW"
                green_alloc = compute_green_time(ns if green_dir == "NS" else ew,
                                                ew if green_dir == "NS" else ns)
                switch_at = now + green_alloc

        time_left = switch_at - now
        if time_left <= 0:
            green_dir = "EW" if green_dir == "NS" else "NS"
            green_alloc = compute_green_time(ns if green_dir == "NS" else ew,
                                            ew if green_dir == "NS" else ns)
            switch_at = now + green_alloc
            time_left = green_alloc

        for b in res.boxes:
            if int(b.cls[0]) not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (255, 255, 0), 2)
        draw_header(frame, ns, ew, green_dir, max(0.0, time_left), green_alloc, p_cong)

        panel = np.zeros((DISP_H, 520, 3), dtype=np.uint8)
        cv2.putText(panel, "Live Graphs", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if len(series_total) >= 2:
            ymax = max(5.0, float(max(series_total)) + 2.0)
        else:
            ymax = 30.0

        plot_series(panel, series_total, 12, 50, 496, 220, 0.0, ymax,
                    "Vehicle Count (smoothed)", (0, 255, 255))
        plot_series(panel, series_p, 12, 290, 496, 220, 0.0, 1.0,
                    "Congestion Probability p(t)", (0, 0, 255))

        cv2.imshow("Traffic Video", frame)
        cv2.imshow("Prediction Graphs", panel)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()