# Face detection guide for Python

The MediaPipe Face Detector task lets you detect faces in an image or video. You can use
this task to locate faces and facial features within a frame. This task uses a
machine learning (ML) model that works with single images or a continuous
stream of images. The task outputs face locations, along with the following
facial key points: left eye, right eye, nose tip, mouth, left eye tragion, and
right eye tragion.

The code sample described in these instructions is available on
[GitHub](https://github.com/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb).
For more information about the capabilities, models, and configuration options
of this task, see the [Overview](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/index).

## Code example

The example code for Face Detector provides a complete implementation of this
task in Python for your reference. This code helps you test this task and get
started on building your own face detector. You can view, run, and
edit the
[Face Detector example code](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb)
using just your web browser.

If you are implementing the Face Detector for Raspberry Pi, refer to the
[Raspberry Pi example
app](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_detector/raspberry_pi).

## Setup

This section describes key steps for setting up your development environment and
code projects specifically to use Face Detector. For general information on
setting up your development environment for using MediaPipe tasks, including
platform version requirements, see the
[Setup guide for Python](https://ai.google.dev/mediapipe/solutions/setup_python).

> [!WARNING]
> **Attention:** This MediaPipe Solutions Preview is an early release. [Learn more](https://ai.google.dev/edge/mediapipe/solutions/about#notice).

### Packages

The MediaPipe Face Detector task requires the mediapipe PyPI package.
You can install and import these dependencies with the following:

    $ python -m pip install mediapipe

### Imports

Import the following classes to access the Face Detector task functions:

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

### Model

The MediaPipe Face Detector task requires a trained model that is compatible with this
task. For more information on available trained models for Face Detector, see
the task overview [Models section](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/index#models).

Select and download the model, and then store it in a local directory:

    model_path = '/absolute/path/to/face_detector.task'

Use the `BaseOptions` object `model_asset_path` parameter to specify the path
of the model to use. For a code example, see the next section.

## Create the task

The MediaPipe Face Detector task uses the `create_from_options` function to
set up the task. The `create_from_options` function accepts values
for configuration options to handle. For more information on configuration
options, see [Configuration options](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python#configuration_options).

The following code demonstrates how to build and configure this task.

These samples also show the variations of the task construction for images,
video files, and live stream.

### Image

```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.IMAGE)
with FaceDetector.create_from_options(options) as detector:
  # The detector is initialized. Use it here.
  # ...
    
```

### Video

```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.VIDEO)
with FaceDetector.create_from_options(options) as detector:
  # The detector is initialized. Use it here.
  # ...
    
```

### Live stream

```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the live stream mode:
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    print('face detector result: {}'.format(result))

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with FaceDetector.create_from_options(options) as detector:
  # The detector is initialized. Use it here.
  # ...
    
```

> [!NOTE]
> **Note:** If you use the video mode or live stream mode, Face Detector uses tracking to avoid triggering the detection model on every frame, which helps reduce latency.

For a complete example of creating a Face Detector for use with an image, see the
[code example](https://colab.sandbox.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb#scrollTo=Iy4r2_ePylIa).

### Configuration options

This task has the following configuration options for Python applications:

| Option Name | Description | Value Range | Default Value |
|---|---|---|---|
| `running_mode` | Sets the running mode for the task. There are three modes: <br /> IMAGE: The mode for single image inputs. <br /> VIDEO: The mode for decoded frames of a video. <br /> LIVE_STREAM: The mode for a livestream of input data, such as from a camera. In this mode, resultListener must be called to set up a listener to receive results asynchronously. | {`IMAGE, VIDEO, LIVE_STREAM`} | `IMAGE` |
| `min_detection_confidence` | The minimum confidence score for the face detection to be considered successful. | `Float [0,1]` | `0.5` |
| `min_suppression_threshold` | The minimum non-maximum-suppression threshold for face detection to be considered overlapped. | `Float [0,1]` | `0.3` |
| `result_callback` | Sets the result listener to receive the detection results asynchronously when the Face Detector is in the live stream mode. Can only be used when running mode is set to `LIVE_STREAM`. | `N/A` | `Not set` |

## Prepare data

Prepare your input as an image file or a numpy array,
then convert it to a `mediapipe.Image` object. If your input is a video file
or live stream from a webcam, you can use an external library such as
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

The Face Detector uses the `detect`, `detect_for_video` and `detect_async`
functions to trigger inferences. For face detection, this involves
preprocessing input data and detecting faces in the image.

The following code demonstrates how to execute the processing with the task model.

### Image

```python
# Perform face detection on the provided single image.
# The face detector must be created with the image mode.
face_detector_result = detector.detect(mp_image)
    
```

### Video

```python
# Perform face detection on the provided single image.
# The face detector must be created with the video mode.
face_detector_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
```

### Live stream

```python
# Send live image data to perform face detection.
# The results are accessible via the `result_callback` provided in
# the `FaceDetectorOptions` object.
# The face detector must be created with the live stream mode.
detector.detect_async(mp_image, frame_timestamp_ms)
    
```

Note the following:

- When running in the video mode or the live stream mode, also provide the Face Detector task the timestamp of the input frame.
- When running in the image or the video model, the Face Detector task blocks the current thread until it finishes processing the input image or frame.
- When running in the live stream mode, the Face Detector task returns immediately and doesn't block the current thread. It will invoke the result listener with the detection result every time it finishes processing an input frame. If the detection function is called when the Face Detector task is busy processing another frame, the task will ignore the new input frame.

For a complete example of running an Face Detector on an image, see the
[code example](https://colab.sandbox.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb#scrollTo=Iy4r2_ePylIa)
for details.

## Handle and display results

The Face Detector returns a `FaceDetectorResult` object for each detection
run. The result object contains bounding boxes for the detected faces and a
confidence score for each detected face.

The following shows an example of the output data from this task:

    FaceDetectionResult:
      Detections:
        Detection #0:
          BoundingBox:
            origin_x: 126
            origin_y: 100
            width: 463
            height: 463
          Categories:
            Category #0:
              index: 0
              score: 0.9729152917861938
          NormalizedKeypoints:
            NormalizedKeypoint #0:
              x: 0.18298381567001343
              y: 0.2961040139198303
            NormalizedKeypoint #1:
              x: 0.3302789330482483
              y: 0.29289937019348145
            ... (6 keypoints for each face)
        Detection #1:
          BoundingBox:
            origin_x: 616
            origin_y: 193
            width: 430
            height: 430
          Categories:
            Category #0:
              index: 0
              score: 0.9251380562782288
          NormalizedKeypoints:
            NormalizedKeypoint #0:
              x: 0.6151331663131714
              y: 0.3713381886482239
            NormalizedKeypoint #1:
              x: 0.7460576295852661
              y: 0.38825345039367676
            ... (6 keypoints for each face)

The following image shows a visualization of the task output:

![Two children with bounding boxes around their faces](https://ai.google.dev/static/mediapipe/images/solutions/face-detection-output.png)

For the image without bounding boxes, see the [original image](https://pixabay.com/photos/brother-sister-girl-family-boy-977170/).

The Face Detector example code demonstrates how to display the
results returned from the task, see the
[code example](https://colab.sandbox.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb#scrollTo=Iy4r2_ePylIa).
for details.