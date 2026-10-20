# Head Gesture Classifier

Recognises head gestures from about 1 second (31 frames) of MediaPipe Face Mesh landmarks.
It was trained on the **CCDb-HG** dataset, which contains natural head gestures from real
conversations rather than acted ones.

## Gestures

| ID | Gesture | What the head does |
|---|---|---|
| 0 | **None** | No gesture: the head is still, or moves without a gesture pattern |
| 1 | **Nod** | Moves up and down repeatedly ("yes") |
| 2 | **Shake** | Rotates left and right repeatedly ("no") |
| 3 | **Tilt** | Leans sideways towards a shoulder (roll) |
| 4 | **Turn** | Turns once to the left or right (single yaw movement) |
| 5 | **Up/Down** | Raises or lowers once (single pitch movement, not repeated like a nod) |

The labels are in [`head_gesture_classifier_label.csv`](head_gesture_classifier_label.csv), in ID order.

## Results

Evaluated on **4 people the model never saw in training** (P2, P5, P10, P20, the same test
subjects as the dataset authors). Each sample is a 31-frame window, labelled with the gesture at its centre frame.

| Gesture | Precision | Recall | F1 | Test windows |
|---|---|---|---|---|
| None | 0.96 | 0.81 | 0.88 | 27,737 |
| Nod | 0.65 | 0.81 | 0.72 | 4,810 |
| Shake | 0.69 | 0.69 | 0.69 | 1,231 |
| Tilt | 0.28 | 0.79 | 0.41 | 628 |
| Turn | 0.25 | 0.80 | 0.38 | 661 |
| Up/Down | 0.16 | 0.31 | 0.21 | 312 |

**Accuracy 0.81 · macro F1 0.55** (0.48 over the five gestures). The exported TFLite model gives the same scores.

What this means in practice:
- **Nod** and **Shake** are reliable.
- **Tilt** and **Turn** are usually caught (recall about 0.8), but they also fire on ordinary head movement (low precision).
  If you get too many false alarms, raise `score_th` in `HeadGestureClassifier`.
- **Up/Down** is weak. It's the rarest class in the data and is easily confused with a nod or ordinary movement.

The gestures in this data are subtle, natural ones from conversation. Deliberate, exaggerated
gestures in front of a webcam are generally easier to recognise.

## Files

| File | Description |
|---|---|
| `head_gesture_classifier.tflite` | Model used for inference (73 KB) |
| `head_gesture_classifier.keras` | Keras model, for retraining or fine-tuning |
| `head_gesture_classifier.py` | `HeadGestureClassifier`, plus landmark selection and pre-processing |
| `head_gesture_classifier_label.csv` | Gesture names |
| `metrics.json` | Full validation and test metrics, including confusion matrices |
| [`../../head_gesture_training.py`](../../head_gesture_training.py) | Training script |

## How it works

1. **Face Mesh** (`mp.solutions.face_mesh`) gives 468 landmarks per frame.
2. Five points are computed by averaging groups of landmarks: **left eye, right eye, nose, left ear, right ear** (`select_face_landmarks`).
3. A rolling window of **31 frames** (about 1 s at 30 fps) is normalised (`pre_process_head_history`):
   - divided by the average ear-to-ear distance, so it works at any distance from the camera;
   - the window's average position is subtracted, so only the **motion** remains.
4. A small 1D CNN classifies the 31 × 15 features into the 6 gestures.

## Usage

```python
from collections import deque
import csv

import cv2 as cv
import mediapipe as mp

from model import HeadGestureClassifier
from model.head_gesture_classifier.head_gesture_classifier import (
    WINDOW_SIZE, select_face_landmarks, pre_process_head_history)

with open('model/head_gesture_classifier/head_gesture_classifier_label.csv', encoding='utf-8-sig') as f:
    labels = [row[0] for row in csv.reader(f)]

classifier = HeadGestureClassifier()
face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
history = deque(maxlen=WINDOW_SIZE)

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        landmarks = [[p.x * w, p.y * h, p.z * w] for p in results.multi_face_landmarks[0].landmark]
        history.append(select_face_landmarks(landmarks))
    else:
        history.clear()

    if len(history) == WINDOW_SIZE:
        gesture = labels[classifier(pre_process_head_history(history))]
        cv.putText(frame, gesture, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv.imshow('Head Gesture', frame)
    if cv.waitKey(1) == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
```

Notes:
- The model was trained on **30 fps** video. At a much lower webcam frame rate, a 31-frame window covers more time than it did in training, and accuracy drops.
- Each prediction describes the **centre** of the window, so results lag by about 0.5 s.

## Retraining

1. Download the features file (209 MB) from the CCDb-HG repository:
   `data/extraction/samples_video_hp_mediapipe_v1_g_xgaze_with_FaceAlignPnPHeadPose_lm_mediapipe_v1_selected.p`
2. Run:

```bash
python head_gesture_training.py --data path/to/that_file.p
```

Training takes about 3 minutes on a CPU. Details:
- **Split:** 16 people for training, 2 for validation (P3, P15), 4 for testing.
- **Augmentation:** mirrored copies of every training window.
- **Sampling:** gesture frames are sampled more densely than "None" frames.
- **Class weights:** none of the other measures stops "None" from dominating, so classes are also weighted.

Adding the dataset's head-pose angles as extra inputs improved test macro F1 by only 0.01,
so the model uses landmarks only and needs no extra MediaPipe model.

## Dataset, credit and licence

Trained on **CCDb-HG**, the head gesture annotations of the Cardiff Conversation Database:

> P. Vuillecard, A. Farkhondeh, M. Villamizar, J.-M. Odobez. *CCDb-HG: Novel Annotations and Gaze-Aware
> Representations for Head Gesture Recognition.* IEEE Face and Gesture (FG), 2024.
> Code and features: <https://github.com/idiap/ccdbhg-head-gesture-recognition> ·
> Annotations: <https://zenodo.org/records/13927536>

The landmark selection and the 31-frame window follow that work. The model, training code
and pre-processing in this folder are an independent re-implementation in TensorFlow.

The dataset is licensed under **CC BY-NC-SA 4.0**. The trained model files (`.tflite`, `.keras`) are
derived from it, so they are also **CC BY-NC-SA 4.0**:
- **non-commercial use only**;
- you must credit the dataset authors;
- anything derived from these files must be shared under the same licence.
