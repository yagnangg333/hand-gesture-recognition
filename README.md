# Hand Gesture Recognition using MediaPipe

A real-time hand gesture recognition system built with [MediaPipe](https://mediapipe.dev/) and [OpenCV](https://opencv.org/).  
It detects hands, tracks 21 landmarks per hand, and classifies gestures based on landmark positions.

## Features
- 🎯 Real-time hand tracking and gesture detection
- ✋ 21 hand landmarks extracted using MediaPipe Hands
- 🤖 Easy to customize with your own gestures
- 🚀 Lightweight and fast — works on CPU
- 📷 Live webcam support (can be extended to videos/images)

## Recognised Signs

### Hand signs: `model/keypoint_classifier`
| ID | Sign |
|---|---|
| 0 | Open (open palm) |
| 1 | Close (fist) |
| 2 | Pointer (index finger up) |
| 3 | OK |
| 4 | Peace sign |

### Finger gestures: `model/point_history_classifier`
Movement of the index fingertip, tracked while the hand shows **Pointer**.

| ID | Gesture |
|---|---|
| 0 | Stop (no movement) |
| 1 | Clockwise (circle) |
| 2 | Counter Clockwise (circle) |
| 3 | Move |

### Head gestures: `model/head_gesture_classifier`
Trained on the CCDb-HG dataset (CC BY-NC-SA 4.0, non-commercial).
See the [head gesture README](model/head_gesture_classifier/README.md) for accuracy, usage and licence.

| ID | Gesture | What the head does |
|---|---|---|
| 0 | None | No gesture |
| 1 | Nod | Moves up and down repeatedly ("yes") |
| 2 | Shake | Rotates left and right repeatedly ("no") |
| 3 | Tilt | Leans sideways towards a shoulder |
| 4 | Turn | Turns once to the left or right |
| 5 | Up/Down | Raises or lowers once |

## How It Works
1. **Hand Detection**: Detects hands in the video feed using MediaPipe’s palm detector.
2. **Landmark Extraction**: Extracts 21 3D landmarks for each detected hand.
3. **Gesture Recognition**: Classifies the gesture based on the spatial relationship between landmarks.
4. **Action Mapping**: Trigger actions or events based on the recognized gesture.

## Technologies Used
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [OpenCV-Python](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [TensorFlow Lite](https://www.tensorflow.org/lite) (gesture classifiers)
- [Streamlit](https://streamlit.io/) (web UI)

## Setup Instructions
Requires **Python 3.9 – 3.12**. The dependencies are pinned in `requirements.txt` because
newer MediaPipe releases removed the `mp.solutions` API this project uses.

```bash
# Clone the repository
git clone https://github.com/yagnangg333/hand-gesture-recognition.git
cd hand-gesture-recognition

# Create a virtual environment (Windows: py -3.12 -m venv .venv  then  .venv\Scripts\activate)
python3.12 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

> **Windows:** if `pip install` fails with `OSError: [Errno 2] No such file or directory ... tensorflow\include\...`,
> the path is too long for Windows. Either [enable long paths](https://pip.pypa.io/warnings/enable-long-paths)
> or create the virtual environment at a short path, e.g. `py -3.12 -m venv D:\venvs\hgr`.
> If activation is blocked by PowerShell, skip it and call the venv's Python directly:
> `D:\venvs\hgr\Scripts\python.exe -m pip install -r requirements.txt` and
> `D:\venvs\hgr\Scripts\python.exe -m streamlit run app.py`.

## Running
| Command | What it does |
|---|---|
| `streamlit run app.py` | Full app: hand-sign + finger-gesture classification, up to 2 hands |
| `streamlit run test.py` | Same as `app.py`, 1 hand |
| `streamlit run streamlit-app.py` | Minimal demo: landmark drawing only, no classification |

In `app.py` / `test.py`, click **Start Recognition** in the sidebar to open the webcam and **Stop** to release it.
Changing a setting while running restarts the camera with the new values.

### Screenshots
![Screenshot 2025-04-28 133346](https://github.com/user-attachments/assets/32050714-1c7e-4229-ab98-5cc088d2d5d3)

![Screenshot 2025-04-28 133359](https://github.com/user-attachments/assets/d659ac3b-35c3-4d5d-9a66-7ffbba76ad4f)

![Screenshot 2025-04-28 133423](https://github.com/user-attachments/assets/46e73efb-570a-4dfb-b040-c1e4ef261b37)

### Collecting training data
Use the **Data Logging** section of the sidebar (instead of the original `k`/`h`/`0-9` keys):
choose *Logging Key Point* or *Logging Point History* and a **Class ID**. While logging is on, every
frame with a detected hand is appended to `model/keypoint_classifier/keypoint.csv` or
`model/point_history_classifier/point_history.csv`. Set the mode back to *Off* when done, then retrain with
`keypoint_classification_EN.ipynb` / `point_history_classification.ipynb`.
