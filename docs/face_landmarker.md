# Face landmark detection guide for Python

The MediaPipe Face Landmarker task lets you detect face landmarks and facial expressions in
images and videos. You can use this task to identify human facial expressions
and apply facial filters and effects to create a virtual avatar. This task uses
machine learning (ML) models that can work with single images or a continuous
stream of images. The task outputs 3-dimensional face landmarks, blendshape
scores (coefficients representing facial expression) to infer detailed facial
surfaces in real-time, and transformation matrices to perform the
transformations required for effects rendering.

The code sample described in these instructions is available on
[GitHub](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_landmarker/python).
For more information about the capabilities, models, and configuration options
of this task, see the [Overview](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index).

## Code example

The example code for Face Landmarker provides a complete implementation of this
task in Python for your reference. This code helps you test this task and get
started on building your own face landmarker. You can view, run, and edit the [Face Landmarker example
code](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb)
using just your web browser.

If you are implementing the Face Landmarker for Raspberry Pi, refer to the
[Raspberry Pi example
app](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_landmarker/raspberry_pi).

## Setup

This section describes key steps for setting up your development environment and
code projects specifically to use Face Landmarker. For general information on
setting up your development environment for using MediaPipe tasks, including
platform version requirements, see the [Setup guide for
Python](https://ai.google.dev/mediapipe/solutions/setup_python).

> [!WARNING]
> **Attention:** This MediaPipe Solutions Preview is an early release. [Learn more](https://ai.google.dev/edge/mediapipe/solutions/about#notice).

### Packages

The MediaPipe Face Landmarker task requires the mediapipe PyPI package. You can install and
import these dependencies with the following:

    $ python -m pip install mediapipe

### Imports

Import the following classes to access the Face Landmarker task functions:

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

### Model

The MediaPipe Face Landmarker task requires a trained model that is compatible with this
task. For more information on available trained models for Face Landmarker, see
the task overview [Models section](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models).

Select and download the model, and then store it in a local directory:

    model_path = '/absolute/path/to/face_landmarker.task'

Use the `BaseOptions` object `model_asset_path` parameter to specify the path of
the model to use. For a code example, see the next section.

## Create the task

The MediaPipe Face Landmarker task uses the `create_from_options` function to set up the
task. The `create_from_options` function accepts values for configuration
options to handle. For more information on configuration options, see
[Configuration options](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python#configuration_options).

The following code demonstrates how to build and configure this task.

These samples also show the variations of the task construction for images,
video files, and live stream.

### Image

```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    
```

### Video

```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the video mode:
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    
```

### Live stream

```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    
```

> [!NOTE]
> **Note:** If you use the video mode or live stream mode, Face Landmarker uses tracking to avoid triggering the model on every frame, which helps reduce latency.

For a complete example of creating a Face Landmarker for use with an image, see
the [code
example](https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb).

### Configuration options

This task has the following configuration options for Python applications:

| Option Name | Description | Value Range | Default Value |
|---|---|---|---|
| `running_mode` | Sets the running mode for the task. There are three modes: <br /> IMAGE: The mode for single image inputs. <br /> VIDEO: The mode for decoded frames of a video. <br /> LIVE_STREAM: The mode for a livestream of input data, such as from a camera. In this mode, resultListener must be called to set up a listener to receive results asynchronously. | {`IMAGE, VIDEO, LIVE_STREAM`} | `IMAGE` |
| `num_faces` | The maximum number of faces that can be detected by the the `FaceLandmarker`. Smoothing is only applied when `num_faces` is set to 1. | `Integer > 0` | `1` |
| `min_face_detection_confidence` | The minimum confidence score for the face detection to be considered successful. | `Float [0.0,1.0]` | `0.5` |
| `min_face_presence_confidence` | The minimum confidence score of face presence score in the face landmark detection. | `Float [0.0,1.0]` | `0.5` |
| `min_tracking_confidence` | The minimum confidence score for the face tracking to be considered successful. | `Float [0.0,1.0]` | `0.5` |
| `output_face_blendshapes` | Whether Face Landmarker outputs face blendshapes. Face blendshapes are used for rendering the 3D face model. | `Boolean` | `False` |
| `output_facial_transformation_matrixes` | Whether FaceLandmarker outputs the facial transformation matrix. FaceLandmarker uses the matrix to transform the face landmarks from a canonical face model to the detected face, so users can apply effects on the detected landmarks. | `Boolean` | `False` |
| `result_callback` | Sets the result listener to receive the landmarker results asynchronously when FaceLandmarker is in the live stream mode. Can only be used when running mode is set to `LIVE_STREAM` | `ResultListener` | `N/A` |

## Prepare data

Prepare your input as an image file or a numpy array, then convert it to a
`mediapipe.Image` object. If your input is a video file or live stream from a
webcam, you can use an external library such as
[OpenCV](https://github.com/opencv/opencv) to load your input frames as numpy
arrays.

### Image

```python
import mediapipe as mp

# Load the input image from an image file.
mp_image = mp.Image.create_from_file('/path/to/image')

# Load the input image from a numpy array.
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    
```

### Video

```python
import mediapipe as mp

# Use OpenCV's VideoCapture to load the input video.

# Load the frame rate of the video using OpenCV's CV_CAP_PROP_FPS
# You'll need it to calculate the timestamp for each frame.

# Loop through each frame in the video using VideoCapture#read()

# Convert the frame received from OpenCV to a MediaPipe's Image object.
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
    
```

### Live stream

```python
import mediapipe as mp

# Use OpenCV's VideoCapture to start capturing from the webcam.

# Create a loop to read the latest frame from the camera using VideoCapture#read()

# Convert the frame received from OpenCV to a MediaPipe's Image object.
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
    
```

## Run the task

The Face Landmarker uses the `detect`, `detect_for_video` and `detect_async`
functions to trigger inferences. For face landmarking, this involves
preprocessing input data and detecting faces in the image.

The following code demonstrates how to execute the processing with the task
model.

### Image

```python
# Perform face landmarking on the provided single image.
# The face landmarker must be created with the image mode.
face_landmarker_result = landmarker.detect(mp_image)
    
```

### Video

```python
# Perform face landmarking on the provided single image.
# The face landmarker must be created with the video mode.
face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    
```

### Live stream

```python
# Send live image data to perform face landmarking.
# The results are accessible via the `result_callback` provided in
# the `FaceLandmarkerOptions` object.
# The face landmarker must be created with the live stream mode.
landmarker.detect_async(mp_image, frame_timestamp_ms)
    
```

Note the following:

- When running in the video mode or the live stream mode, also provide the Face Landmarker task the timestamp of the input frame.
- When running in the image or the video model, the Face Landmarker task blocks the current thread until it finishes processing the input image or frame.
- When running in the live stream mode, the Face Landmarker task returns immediately and doesn't block the current thread. It will invoke the result listener with the detection result every time it finishes processing an input frame. If the detection function is called when the Face Landmarker task is busy processing another frame, the task will ignore the new input frame.

For a complete example of running an Face Landmarker on an image, see the [code
example](https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb)
for details.

## Handle and display results

The Face Landmarker returns a `FaceLandmarkerResult` object for each detection
run. The result object contains a face mesh for each detected face, with
coordinates for each face landmark. Optionally, the result object can also
contain blendshapes, which denote facial expressions, and a facial
transformation matrix to apply face effects on the detected landmarks.

The following shows an example of the output data from this task:

    FaceLandmarkerResult:
      face_landmarks:
        NormalizedLandmark #0:
          x: 0.5971359014511108
          y: 0.485361784696579
          z: -0.038440968841314316
        NormalizedLandmark #1:
          x: 0.3302789330482483
          y: 0.29289937019348145
          z: -0.09489090740680695
        ... (478 landmarks for each face)
      face_blendshapes:
        browDownLeft: 0.8296722769737244
        browDownRight: 0.8096957206726074
        browInnerUp: 0.00035583582939580083
        browOuterUpLeft: 0.00035752105759456754
        ... (52 blendshapes for each face)
      facial_transformation_matrixes:
        [9.99158978e-01, -1.23036895e-02, 3.91213447e-02, -3.70770246e-01]
        [1.66496094e-02,  9.93480563e-01, -1.12779640e-01, 2.27719707e+01]
        ...

The following image shows a visualization of the task output:

![A man with the regions of his face geometrically mapped out to indicate his face's shape and dimensions](https://ai.google.dev/static/mediapipe/images/solutions/face_landmarker_output.png)

The Face Landmarker example code demonstrates how to display the results returned
from the task, see the [code
example](https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb)
for details.