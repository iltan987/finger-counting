This page shows you how to set up your development environment to use MediaPipe
Tasks in your Python applications.

## Supported platforms and versions

Building applications with MediaPipe Tasks requires the following development
environment resources:

- OS:
  - Desktop: Windows, Mac, Linux
  - IoT: Raspberry OS 64-bit
- Python: Supports version 3.9 and later versions
- PIP: version 20.3+

> [!NOTE]
> **Note:** If you are testing or experimenting with MediaPipe tasks, consider using [Colaboratory](https://colab.research.google.com/), a Python notebook environment that requires no setup and runs entirely in the cloud.

## Developer environment setup

Before running a MediaPipe task on a Python application, install the MediaPipe
package.

    $ python -m pip install mediapipe

> [!WARNING]
> **Attention:** This MediaPipe Solutions Preview is an early release. [Learn more](https://ai.google.dev/edge/mediapipe/solutions/about#notice).

After installing the package, import it into your development project.

    import mediapipe as mp

## MediaPipe Tasks dependencies

MediaPipe Tasks provides three prebuilt libraries for vision, text, audio.
Depending on the MediaPipe Task used by the app, import the vision, text, or
audio library into your development project.

### Vision tasks

The MediaPipe Tasks vision module contains tasks that handle image or video
inputs. To import the MediaPipe Tasks vision library, import the following
dependency to your into your development project.

    from mediapipe.tasks.python import vision

### Text tasks

The MediaPipe Tasks text module contains tasks that handle string inputs.To
import the MediaPipe Tasks text library, import the following dependency to your
into your development project.

    from mediapipe.tasks.python import text

### Audio tasks

The MediaPipe Tasks audio module contains tasks that handle sound inputs. To
import the MediaPipe Tasks audio library, import the following dependency to
your into your development project.

    from mediapipe.tasks.python import audio

## BaseOptions configuration

The BaseOptions allow for general configuration of MediaPipe Task APIs.

| Option name | Description | Accepted values |
|---|---|---|
| `model_asset_buffer` | The model asset file contents. | Model content as a byte string |
| `model_asset_path` | The path of the model asset to open and map into memory. | File path as a string |

## Packaging Python Tasks apps with PyInstaller

When packaging a Python Tasks app with PyInstaller, model files such as
`.task` bundles are not included automatically. If your app uses
`BaseOptions(model_asset_path=...)`, include the model file in the PyInstaller
bundle and resolve its path at runtime.

For example, use a helper that checks for PyInstaller's `sys._MEIPASS`
directory:

    import os
    import sys

    def resource_path(relative_path: str) -> str:
        base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
        return os.path.join(base_path, relative_path)

Then pass the resolved model path to `BaseOptions`:

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = resource_path("pose_landmarker.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_options)
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

Bundle the model file and collect MediaPipe package files when building with
PyInstaller:

    pyinstaller app.py \
      --add-data "pose_landmarker.task:." \
      --collect-all mediapipe \
      --hidden-import mediapipe.tasks.c

On macOS or Linux, the `--add-data` separator is `:`. On Windows, use `;`. And for macOS desktop apps, add `--windowed` to create an `.app` bundle:

## Troubleshooting

For help with technical questions related to MediaPipe, visit the
[discussion group](https://groups.google.com/forum/#!forum/mediapipe) or
[Stack Overflow](https://stackoverflow.com/questions/tagged/mediapipe) for
support from the community. To report bugs or make feature requests,
[file an issue on GitHub](https://github.com/google/mediapipe/issues).

For help setting up your Python development environment, visit the
[Python developer's guide](https://devguide.python.org/).