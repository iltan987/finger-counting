import math
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "gesture_recognizer.task"

# (MCP, PIP, TIP) triplets for index / middle / ring / pinky.
FINGER_JOINTS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
# (MCP, IP, TIP) for the thumb.
THUMB_JOINTS = (2, 3, 4)

# A finger is "extended" when the angle at its middle joint is close to 180°.
FINGER_ANGLE_THRESHOLD = 160.0
THUMB_ANGLE_THRESHOLD = 150.0  # thumb is anatomically less straight

# Majority-vote window for displayed counts (kills flicker on borderline poses).
SMOOTHING_WINDOW = 5


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print(f"Downloading gesture recognizer model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


def _joint_angle(a, b, c) -> float:
    # Angle at vertex b formed by points a-b-c, in degrees. Uses 3D landmark
    # coords so it's robust to hand rotation / tilt.
    bax, bay, baz = a.x - b.x, a.y - b.y, a.z - b.z
    bcx, bcy, bcz = c.x - b.x, c.y - b.y, c.z - b.z
    dot = bax * bcx + bay * bcy + baz * bcz
    mag = math.sqrt((bax * bax + bay * bay + baz * baz) *
                    (bcx * bcx + bcy * bcy + bcz * bcz))
    if mag == 0.0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _mode(values) -> int:
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]


def count_fingers(landmarks) -> int:
    count = 0
    for mcp, pip, tip in FINGER_JOINTS:
        if _joint_angle(landmarks[mcp], landmarks[pip], landmarks[tip]) > FINGER_ANGLE_THRESHOLD:
            count += 1
    mcp, ip, tip = THUMB_JOINTS
    if _joint_angle(landmarks[mcp], landmarks[ip], landmarks[tip]) > THUMB_ANGLE_THRESHOLD:
        count += 1
    return count


def draw_hand(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 0), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1, cv2.LINE_AA)


def main() -> None:
    model_path = ensure_model()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    # Force MJPEG: raw YUYV at 640x480x30fps is ~221 Mbps and saturates
    # usbipd's TCP transport (causes V4L2 select() timeouts on WSL2).
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    t0 = time.monotonic()
    last_ts_ms = -1

    prev_loop_t: float | None = None
    fps_ema = 0.0
    infer_ms_ema = 0.0
    EMA_ALPHA = 0.1

    total_history: deque = deque(maxlen=SMOOTHING_WINDOW)
    per_hand_history: dict[str, deque] = {
        "Left": deque(maxlen=SMOOTHING_WINDOW),
        "Right": deque(maxlen=SMOOTHING_WINDOW),
    }

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # detect_for_video requires strictly monotonically-increasing
            # millisecond timestamps.
            ts_ms = int((time.monotonic() - t0) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            t_pre = time.monotonic()
            result = recognizer.recognize_for_video(mp_image, ts_ms)
            infer_ms = (time.monotonic() - t_pre) * 1000.0
            infer_ms_ema = (
                infer_ms if infer_ms_ema == 0.0
                else (1 - EMA_ALPHA) * infer_ms_ema + EMA_ALPHA * infer_ms
            )

            total = 0
            for hand_lms, hand_cats, gest_cats in zip(
                result.hand_landmarks, result.handedness, result.gestures
            ):
                draw_hand(frame, hand_lms)

                n = count_fingers(hand_lms)
                total += n

                # MediaPipe's handedness label refers to the (flipped) image,
                # so it's the opposite of the user's actual hand — swap it.
                if hand_cats:
                    cat = hand_cats[0]
                    label = "Left" if cat.category_name == "Right" else "Right"
                    per_hand_history[label].append(n)
                    n_smooth = _mode(per_hand_history[label])
                    info = f"{label} {cat.score * 100:.0f}%  fingers: {n_smooth}"
                else:
                    info = f"fingers: {n}"

                if gest_cats:
                    g = gest_cats[0]
                    if g.category_name and g.category_name != "None":
                        info += f"  {g.category_name} {g.score * 100:.0f}%"

                h, w = frame.shape[:2]
                wrist = hand_lms[0]
                x = int(wrist.x * w)
                y = int(wrist.y * h) + 30
                cv2.putText(
                    frame,
                    info,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            total_history.append(total)
            total_smooth = _mode(total_history)
            cv2.putText(
                frame,
                f"Total: {total_smooth}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )

            # FPS overlay (top-right). Compute before imshow so this frame
            # reflects the latency we just measured.
            h, w = frame.shape[:2]
            x = w - 200
            cv2.putText(frame, f"FPS:   {fps_ema:5.1f}", (x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Infer: {infer_ms_ema:5.1f} ms", (x, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow("Finger Counting", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            now = time.monotonic()
            if prev_loop_t is not None:
                dt = now - prev_loop_t
                if dt > 0:
                    fps = 1.0 / dt
                    fps_ema = fps if fps_ema == 0.0 else (1 - EMA_ALPHA) * fps_ema + EMA_ALPHA * fps
            prev_loop_t = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
