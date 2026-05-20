import math
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)
GESTURE_MODEL_PATH = Path(__file__).resolve().parent / "gesture_recognizer.task"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"

# Only show a blendshape pill above the face when its confidence clears this
# threshold. Most blendshapes idle below 0.2 on a resting face; 0.4 keeps the
# overlay quiet until the user is clearly expressing something.
FACE_BLENDSHAPE_THRESHOLD = 0.4

# (MCP, PIP, TIP) triplets for index / middle / ring / pinky.
FINGER_JOINTS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
# (CMC, MCP, TIP) for the thumb. The IP joint barely bends — when the
# thumb tucks across the palm (e.g. Victory, Closed_Fist) it pivots at
# CMC/MCP, not IP, so the angle at MCP discriminates extended-vs-tucked
# far better than the angle at IP.
THUMB_JOINTS = (1, 2, 4)

# A finger is "extended" when the angle at its middle joint is close to 180°.
FINGER_ANGLE_THRESHOLD = 160.0
THUMB_ANGLE_THRESHOLD = 150.0

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

# EMA factor used by the FPS / inference-latency overlays.
_EMA_ALPHA = 0.1

# Hand-overlay text styling.
_HAND_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HAND_TEXT_SCALE = 0.6
_HAND_TEXT_THICKNESS = 2
_HAND_TEXT_LINE_SPACING = 8
_HAND_TEXT_COLOR = (255, 255, 255)
_GESTURE_PILL_COLOR = (80, 215, 255)   # warm gold
_WRIST_TEXT_OFFSET = 25                 # baseline of the first line below/above the wrist
_PILL_PAD = 4                           # mirrors _draw_text_pill's default

# Face-overlay text styling.
_FACE_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FACE_TEXT_SCALE = 0.55
_FACE_TEXT_THICKNESS = 1

# Top-right performance overlay.
_PERF_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PERF_TEXT_SCALE = 0.45
_PERF_TEXT_THICKNESS = 1
_PERF_TEXT_COLOR = (220, 220, 220)
_PERF_MARGIN = 8

# Generic edge padding used when clamping overlays inside the frame.
_SCREEN_MARGIN = 5


def ensure_model(url: str, path: Path, label: str) -> Path:
    if not path.exists():
        print(f"Downloading {label} model to {path} ...")
        urllib.request.urlretrieve(url, path)
        print("Done.")
    return path


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


# Per-finger colors (BGR). Palm gets a neutral grey; fingers get distinct hues
# so the skeleton reads at a glance.
_HC = vision.HandLandmarksConnections
_HAND_CONNECTION_GROUPS = (
    (_HC.HAND_PALM_CONNECTIONS,         (210, 210, 210)),
    (_HC.HAND_THUMB_CONNECTIONS,        ( 60, 180, 255)),  # orange
    (_HC.HAND_INDEX_FINGER_CONNECTIONS, ( 80, 255,  80)),  # green
    (_HC.HAND_MIDDLE_FINGER_CONNECTIONS,(255, 220,  80)),  # cyan
    (_HC.HAND_RING_FINGER_CONNECTIONS,  (255, 120,  80)),  # blue
    (_HC.HAND_PINKY_FINGER_CONNECTIONS, (220,  80, 255)),  # magenta
)


def draw_hand(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conns, color in _HAND_CONNECTION_GROUPS:
        for conn in conns:
            cv2.line(frame, pts[conn.start], pts[conn.end], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)


# Face mesh palette. Skipping FACE_LANDMARKS_TESSELATION on purpose — the full
# ~2900-edge mesh reads as visual noise over a webcam feed. Contours + irises
# give a clean wireframe. Monochrome cyan/grey HUD look: neutral and tech-y
# rather than makeup-y.
_FC = vision.FaceLandmarksConnections
_FACE_OUTLINE_COLOR = (200, 200, 200)   # light grey
_FACE_FEATURE_COLOR = (220, 200, 140)   # muted cyan-ish
_FACE_IRIS_COLOR    = (255, 240, 200)   # brighter cyan accent
_FACE_CONNECTION_GROUPS = (
    (_FC.FACE_LANDMARKS_FACE_OVAL,     _FACE_OUTLINE_COLOR),
    (_FC.FACE_LANDMARKS_LIPS,          _FACE_FEATURE_COLOR),
    (_FC.FACE_LANDMARKS_LEFT_EYE,      _FACE_FEATURE_COLOR),
    (_FC.FACE_LANDMARKS_LEFT_EYEBROW,  _FACE_FEATURE_COLOR),
    (_FC.FACE_LANDMARKS_RIGHT_EYE,     _FACE_FEATURE_COLOR),
    (_FC.FACE_LANDMARKS_RIGHT_EYEBROW, _FACE_FEATURE_COLOR),
    (_FC.FACE_LANDMARKS_LEFT_IRIS,     _FACE_IRIS_COLOR),
    (_FC.FACE_LANDMARKS_RIGHT_IRIS,    _FACE_IRIS_COLOR),
)


def draw_face(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conns, color in _FACE_CONNECTION_GROUPS:
        for conn in conns:
            cv2.line(frame, pts[conn.start], pts[conn.end], color, 1, cv2.LINE_AA)


def _draw_text_pill(frame, text, x, y, font, scale, thickness, fg,
                    alpha: float = 0.55, pad: int = 4) -> None:
    # Darken a rectangular ROI behind the text for legibility, then draw text.
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    fh, fw = frame.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - th - pad)
    x1 = min(fw, x + tw + pad)
    y1 = min(fh, y + bl + pad)
    if x1 > x0 and y1 > y0:
        roi = frame[y0:y1, x0:x1]
        roi[:] = (roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    cv2.putText(frame, text, (x, y), font, scale, fg, thickness, cv2.LINE_AA)


def _update_ema(ema: float, value: float) -> float:
    # Treat 0.0 as "uninitialized" so the first reading seeds the average
    # instead of bleeding from a zero baseline.
    if ema == 0.0:
        return value
    return (1.0 - _EMA_ALPHA) * ema + _EMA_ALPHA * value


def _draw_hand_overlay(frame, lines, wrist_x: int, wrist_y: int) -> None:
    # Stack pill lines below the wrist; if they'd overflow the bottom edge,
    # flip to render above instead. The top edge gets a clamp too so a hand
    # high in the frame doesn't push the flipped stack off-screen.
    h, w = frame.shape[:2]
    sizes = [cv2.getTextSize(t, _HAND_TEXT_FONT, _HAND_TEXT_SCALE, _HAND_TEXT_THICKNESS)
             for t, _ in lines]
    baseline_span = (sum(s[0][1] for s in sizes[:-1])
                     + _HAND_TEXT_LINE_SPACING * (len(sizes) - 1))
    first_y = wrist_y + _WRIST_TEXT_OFFSET
    last_pill_bottom = first_y + baseline_span + sizes[-1][1] + _PILL_PAD
    if last_pill_bottom > h - _SCREEN_MARGIN:
        first_y = max(sizes[0][0][1] + _PILL_PAD + _SCREEN_MARGIN,
                      wrist_y - _WRIST_TEXT_OFFSET - baseline_span)
    y = first_y
    for (text, color), ((tw, th), _) in zip(lines, sizes):
        x = max(_SCREEN_MARGIN, min(wrist_x, w - tw - _SCREEN_MARGIN))
        _draw_text_pill(frame, text, x, y,
                        _HAND_TEXT_FONT, _HAND_TEXT_SCALE, _HAND_TEXT_THICKNESS,
                        color)
        y += th + _HAND_TEXT_LINE_SPACING


def _draw_face_blendshape(frame, face_lms, face_bs) -> None:
    # Drop _neutral — it's always the top score on a resting face. The next
    # winner only shows up when it clears FACE_BLENDSHAPE_THRESHOLD, keeping
    # the overlay quiet unless the user is actually expressing something.
    ranked = sorted(
        (c for c in face_bs if c.category_name != "_neutral"),
        key=lambda c: c.score,
        reverse=True,
    )
    if not ranked or ranked[0].score < FACE_BLENDSHAPE_THRESHOLD:
        return
    top = ranked[0]
    label = f"{top.category_name} {top.score * 100:.0f}%"
    h, w = frame.shape[:2]
    forehead = face_lms[10]  # top of the face oval
    (tw, th), _ = cv2.getTextSize(label, _FACE_TEXT_FONT, _FACE_TEXT_SCALE,
                                  _FACE_TEXT_THICKNESS)
    x = max(_SCREEN_MARGIN,
            min(int(forehead.x * w) - tw // 2, w - tw - _SCREEN_MARGIN))
    y = max(th + _SCREEN_MARGIN, int(forehead.y * h) - 12)
    _draw_text_pill(frame, label, x, y,
                    _FACE_TEXT_FONT, _FACE_TEXT_SCALE, _FACE_TEXT_THICKNESS,
                    _FACE_IRIS_COLOR)


def _draw_perf_overlay(frame, fps: float, infer_ms: float, face_infer_ms: float) -> None:
    w = frame.shape[1]
    for i, text in enumerate((
        f"FPS:   {fps:5.1f}",
        f"Infer: {infer_ms:5.1f} ms",
        f"Face:  {face_infer_ms:5.1f} ms",
    )):
        (tw, th), _ = cv2.getTextSize(text, _PERF_TEXT_FONT, _PERF_TEXT_SCALE,
                                      _PERF_TEXT_THICKNESS)
        y = _PERF_MARGIN + th + i * (th + 6)
        _draw_text_pill(frame, text, w - tw - _PERF_MARGIN, y,
                        _PERF_TEXT_FONT, _PERF_TEXT_SCALE, _PERF_TEXT_THICKNESS,
                        _PERF_TEXT_COLOR)


def main() -> None:
    gesture_model_path = ensure_model(
        GESTURE_MODEL_URL, GESTURE_MODEL_PATH, "gesture recognizer"
    )
    face_model_path = ensure_model(
        FACE_MODEL_URL, FACE_MODEL_PATH, "face landmarker"
    )

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
            model_asset_path=str(gesture_model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # num_faces=1: built-in temporal smoothing of the 478 landmarks only kicks
    # in at 1, and a single user is enough for this app. Blendshapes give us a
    # facial-expression label parallel to the hand gesture label.
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(face_model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    t0 = time.monotonic()
    last_ts_ms = -1

    prev_loop_t: float | None = None
    fps_ema = 0.0
    infer_ms_ema = 0.0
    face_infer_ms_ema = 0.0

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

    with (
        vision.GestureRecognizer.create_from_options(options) as recognizer,
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
    ):
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
            infer_ms_ema = _update_ema(infer_ms_ema, (time.monotonic() - t_pre) * 1000.0)

            t_pre_face = time.monotonic()
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            face_infer_ms_ema = _update_ema(
                face_infer_ms_ema, (time.monotonic() - t_pre_face) * 1000.0
            )

            t_now = ts_ms / 1000.0
            h, w = frame.shape[:2]

            for hand_lms, hand_cats, gest_cats in zip(
                result.hand_landmarks, result.handedness, result.gestures
            ):
                if not hand_cats:
                    # GestureRecognizer always populates handedness when a hand
                    # passes its detection threshold, so this is essentially
                    # unreachable — but skipping is cheaper than guessing.
                    continue
                cat = hand_cats[0]
                # MediaPipe's handedness label refers to the (flipped) image,
                # so it's the opposite of the user's actual hand — swap it.
                label = "Left" if cat.category_name == "Right" else "Right"

                # Smooth landmarks per hand identity so each hand keeps its own
                # filter state across frames.
                lms = smoothers[label].smooth(hand_lms, t_now)
                draw_hand(frame, lms)

                per_hand_history[label].append(count_fingers(lms))
                n_smooth = _mode(per_hand_history[label])

                lines: list[tuple[str, tuple[int, int, int]]] = [
                    (f"{label} {n_smooth}  ({cat.score * 100:.0f}%)", _HAND_TEXT_COLOR),
                ]

                # Gesture debounce: only show a name after N consecutive
                # matching detections (keyed by the user-perspective hand).
                name = ""
                score = 0.0
                if gest_cats:
                    g = gest_cats[0]
                    if g.category_name and g.category_name != "None":
                        name = g.category_name
                        score = g.score
                history = gesture_history[label]
                history.append(name)
                if (len(history) == GESTURE_DEBOUNCE and name
                        and all(h == name for h in history)):
                    lines.append((f"{name} {score * 100:.0f}%", _GESTURE_PILL_COLOR))

                wrist = lms[0]
                _draw_hand_overlay(frame, lines,
                                   int(wrist.x * w), int(wrist.y * h))

            face_blendshapes = face_result.face_blendshapes
            for i, face_lms in enumerate(face_result.face_landmarks):
                draw_face(frame, face_lms)
                if face_blendshapes and i < len(face_blendshapes):
                    _draw_face_blendshape(frame, face_lms, face_blendshapes[i])

            _draw_perf_overlay(frame, fps_ema, infer_ms_ema, face_infer_ms_ema)

            cv2.imshow("Finger Counting", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            now = time.monotonic()
            if prev_loop_t is not None:
                dt = now - prev_loop_t
                if dt > 0:
                    fps_ema = _update_ema(fps_ema, 1.0 / dt)
            prev_loop_t = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
