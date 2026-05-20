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

# One-Euro filter tunings for landmark coords (normalized [0,1] @ ~30 FPS).
# Higher beta -> snappier on fast motion; lower min_cutoff -> more smoothing
# when the hand is still. Defaults follow MediaPipe's JS hand-tracking demo.
ONE_EURO_MIN_CUTOFF = 1.5
ONE_EURO_BETA = 0.05
ONE_EURO_D_CUTOFF = 1.0

# Require this many consecutive identical detections before showing a gesture.
GESTURE_DEBOUNCE = 3


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


class _OneEuro:
    # https://gery.casiez.net/1euro/ — adaptive low-pass filter.
    __slots__ = ("min_cutoff", "beta", "d_cutoff", "x_prev", "dx_prev", "t_prev")

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0
        self.t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev  # type: ignore[return-value]
        dx = (x - self.x_prev) / dt  # type: ignore[operator]
        a_d = self._alpha(self.d_cutoff, dt)
        dx_s = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_s)
        a = self._alpha(cutoff, dt)
        x_f = a * x + (1 - a) * self.x_prev  # type: ignore[operator]
        self.x_prev = x_f
        self.dx_prev = dx_s
        self.t_prev = t
        return x_f


class _SmoothedLandmark:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class HandSmoother:
    def __init__(self):
        self._fx = [_OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
                    for _ in range(21)]
        self._fy = [_OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
                    for _ in range(21)]
        self._fz = [_OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
                    for _ in range(21)]

    def smooth(self, landmarks, t: float) -> list:
        return [
            _SmoothedLandmark(self._fx[i](lm.x, t),
                              self._fy[i](lm.y, t),
                              self._fz[i](lm.z, t))
            for i, lm in enumerate(landmarks)
        ]


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

    # Staying on CPU + XNNPACK. The MediaPipe GPU delegate works on WSL2
    # once Mesa is pointed at the d3d12 driver (env vars
    # `GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` give
    # us hardware OpenGL ES on the discrete GPU), but for this model the
    # GPU and CPU paths benchmark within 0.2 ms of each other — the graph
    # has CPU-only ops that force per-frame CPU<->GPU syncs.
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    t0 = time.monotonic()
    last_ts_ms = -1

    prev_loop_t: float | None = None
    fps_ema = 0.0
    infer_ms_ema = 0.0
    EMA_ALPHA = 0.1

    per_hand_history: dict[str, deque] = {
        "Left": deque(maxlen=SMOOTHING_WINDOW),
        "Right": deque(maxlen=SMOOTHING_WINDOW),
    }
    smoothers: dict[str, HandSmoother] = {
        "Left": HandSmoother(),
        "Right": HandSmoother(),
    }
    gesture_history: dict[str, deque] = {
        "Left": deque(maxlen=GESTURE_DEBOUNCE),
        "Right": deque(maxlen=GESTURE_DEBOUNCE),
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

            t_now = ts_ms / 1000.0
            for hand_lms, hand_cats, gest_cats in zip(
                result.hand_landmarks, result.handedness, result.gestures
            ):
                # MediaPipe's handedness label refers to the (flipped) image,
                # so it's the opposite of the user's actual hand — swap it.
                label = None
                cat = None
                if hand_cats:
                    cat = hand_cats[0]
                    label = "Left" if cat.category_name == "Right" else "Right"

                # Smooth landmarks per hand identity. Filters are kept across
                # frames so the smoother learns each hand's motion profile.
                if label is not None:
                    lms = smoothers[label].smooth(hand_lms, t_now)
                else:
                    lms = hand_lms

                draw_hand(frame, lms)
                n = count_fingers(lms)

                lines: list[str] = []
                if label is not None and cat is not None:
                    per_hand_history[label].append(n)
                    n_smooth = _mode(per_hand_history[label])
                    lines.append(f"{label} {cat.score * 100:.0f}%  -  {n_smooth}")
                else:
                    lines.append(f"fingers: {n}")

                # Gesture debounce: only show after N consecutive matching
                # detections (keyed by the user-perspective hand).
                if label is not None:
                    name = ""
                    score = 0.0
                    if gest_cats:
                        g = gest_cats[0]
                        if g.category_name and g.category_name != "None":
                            name = g.category_name
                            score = g.score
                    history = gesture_history[label]
                    history.append(name)
                    if len(history) == GESTURE_DEBOUNCE and name and all(h == name for h in history):
                        lines.append(f"{name} {score * 100:.0f}%")

                h, w = frame.shape[:2]
                wrist = lms[0]
                wx = int(wrist.x * w)
                wy = int(wrist.y * h)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thickness = 2
                y = wy + 25
                for line in lines:
                    (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
                    x = max(5, min(wx, w - tw - 5))
                    cv2.putText(frame, line, (x, y), font, scale,
                                (0, 255, 0), thickness, cv2.LINE_AA)
                    y += th + 8

            # FPS overlay, right-aligned to the top edge.
            h, w = frame.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thickness = 1
            margin = 8
            for i, text in enumerate(
                (f"FPS:   {fps_ema:5.1f}", f"Infer: {infer_ms_ema:5.1f} ms")
            ):
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                y = margin + th + i * (th + 6)
                cv2.putText(frame, text, (w - tw - margin, y), font, scale,
                            (200, 200, 200), thickness, cv2.LINE_AA)

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
