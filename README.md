# gesture-mirror

Real-time webcam mirror that draws what your hands and face are doing — finger
counts, canned hand gestures (thumbs-up, peace, OK, etc.), facial expression,
all overlaid on a live feed.

Built on [MediaPipe](https://ai.google.dev/edge/mediapipe) Tasks (GestureRecognizer
+ FaceLandmarker) and OpenCV.

## Features

- **Hands** — per-finger colored skeleton, both hands tracked independently,
  user-perspective left/right labels.
- **Finger counting** — 3D joint-angle method (rotation-invariant), 5-frame
  majority vote so borderline poses don't flicker.
- **Gestures** — `Thumb_Up`, `Thumb_Down`, `Victory`, `Open_Palm`, `Closed_Fist`,
  `Pointing_Up`, `ILoveYou`. 3-frame debounce so transitions don't paint random
  labels.
- **Face mesh** — face oval, eyes, brows, lips, irises. Muted HUD palette
  (skipping the full ~2900-edge tesselation; contours + irises read cleaner).
- **Facial expression** — top non-`_neutral` blendshape above the forehead when
  it clears 40% confidence (smile, blink, brow-raise, etc.).
- **Performance overlay** — FPS plus per-detector inference cost (hand and face)
  in the top-right corner.
- **Landmark smoothing** — adaptive
  [One-Euro filter](https://gery.casiez.net/1euro/) per hand identity to keep
  the mesh stable without adding noticeable lag.

## Prerequisites

- Python 3.13
- A webcam (see [platform notes](#platform-notes) if you're on WSL2)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The two MediaPipe model files (`gesture_recognizer.task`, `face_landmarker.task`,
~12 MB combined) download to the project directory on first run.

### CLI options

```
python main.py [--camera N] [--width N] [--height N] [--fps N]
```

Defaults: camera `0`, `640x480` @ `30` fps. If you have multiple webcams or
want a higher-res feed, override on the command line. `python main.py --help`
for the full list.

### Controls

- `q` or `ESC` — quit

## How it works

`main.py` runs a single video loop that pumps each frame through two MediaPipe
Tasks detectors in series:

1. **`GestureRecognizer`** — returns 21 3D hand landmarks per hand, handedness,
   and a canned gesture name. We replace MediaPipe's "tip-above-pip" finger
   classification with a 3D joint-angle check (`_joint_angle` in `main.py`) —
   the angle is rotation-invariant, so a rotated 5 still counts as 5.
2. **`FaceLandmarker`** — returns 478 3D face landmarks and 52 blendshape
   coefficients (when `output_face_blendshapes=True`). With `num_faces=1`
   MediaPipe also applies its own temporal smoothing internally.

Both models run on CPU via XNNPACK. A GPU delegate path exists, but for models
this small the graph has CPU-only ops that force per-frame CPU↔GPU syncs, so
CPU and GPU benchmark within ~0.2 ms of each other on most setups.

OpenCV picks the platform-appropriate capture backend automatically (V4L2 on
Linux, DirectShow/MSMF on Windows, AVFoundation on macOS). The capture pipe
requests MJPG, which most webcams emit natively — this keeps USB bandwidth
modest at higher resolutions and is friendlier to virtual USB transports.

## Platform notes

### WSL2

If you're running on WSL2 the host Windows machine owns the webcam; forward it
into WSL with [usbipd-win](https://github.com/dorssel/usbipd-win):

```powershell
# in Windows (Admin PowerShell)
usbipd list
usbipd bind --busid <busid>
usbipd attach --wsl --busid <busid>
```

Make sure your WSL user is in the `video` group:

```bash
sudo usermod -aG video $USER
```

then open a fresh shell. The MJPG capture format is essential over usbipd —
the raw YUYV stream saturates its TCP transport and you'll see V4L2
`select() timeout`. `main.py` already forces MJPG, so this just works.

### Wayland

OpenCV's GUI backend uses X11. On a pure Wayland session you'll either need
XWayland (default on most distros) or to swap the display loop for one that
talks Wayland directly — out of scope here.

## License

[MIT](LICENSE)
