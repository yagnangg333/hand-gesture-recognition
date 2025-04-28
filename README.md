# Hand Gesture Recognition using MediaPipe

A real-time hand gesture recognition system built with [MediaPipe](https://mediapipe.dev/) and [OpenCV](https://opencv.org/).  
It detects hands, tracks 21 landmarks per hand, and classifies gestures based on landmark positions.

## Features
- 🎯 Real-time hand tracking and gesture detection
- ✋ 21 hand landmarks extracted using MediaPipe Hands
- 🤖 Easy to customize with your own gestures
- 🚀 Lightweight and fast — works on CPU
- 📷 Live webcam support (can be extended to videos/images)

## How It Works
1. **Hand Detection**: Detects hands in the video feed using MediaPipe’s palm detector.
2. **Landmark Extraction**: Extracts 21 3D landmarks for each detected hand.
3. **Gesture Recognition**: Classifies the gesture based on the spatial relationship between landmarks.
4. **Action Mapping**: Trigger actions or events based on the recognized gesture.

## Technologies Used
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [OpenCV-Python](https://opencv.org/)
- [NumPy](https://numpy.org/)

## Setup Instructions
```bash
# Clone the repository
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition

# Install dependencies
pip install mediapipe opencv-python numpy streamlit

# Run the app
streamlit run app.py
