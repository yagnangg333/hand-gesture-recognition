import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Streamlit UI
st.title("Hand Gesture Recognition")
st.sidebar.title("Settings")

# Webcam Feed
run = st.sidebar.checkbox("Start Webcam", value=False)

if run:
    cap = cv2.VideoCapture(0)  # Capture video from webcam
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    stframe = st.empty()  # Placeholder for video feed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        stframe.image(frame, channels="RGB")  # Display frame in Streamlit

    cap.release()

st.sidebar.info("Click the checkbox to start the webcam.")
