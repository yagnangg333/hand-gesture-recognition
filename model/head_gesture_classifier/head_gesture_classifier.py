#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# Number of frames per sample (~1 s at 30 fps), centred on the classified frame
WINDOW_SIZE = 31

# MediaPipe Face Mesh landmark indices averaged into 5 points
# (same selection as the CCDb-HG dataset: left eye, right eye, nose, left ear, right ear)
LEFT_EYE = [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]
RIGHT_EYE = [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341]
NOSE = [51, 5, 281, 45, 4, 275, 220, 134, 236, 3, 195, 248, 456, 363, 440]
EAR_L = [234, 93, 227, 137]
EAR_R = [454, 323, 447, 366]


def select_face_landmarks(landmarks):
    """468x3 pixel-space Face Mesh landmarks (x*w, y*h, z*w) -> 15 values."""
    landmarks = np.asarray(landmarks, dtype=np.float64)
    points = [landmarks[idx].mean(axis=0) for idx in (LEFT_EYE, RIGHT_EYE, NOSE, EAR_L, EAR_R)]
    return np.concatenate(points)


def pre_process_head_history(landmark_history):
    """WINDOW_SIZE x 15 selected landmarks -> flat list of WINDOW_SIZE * 15 motion features"""
    landmarks = np.asarray(landmark_history, dtype=np.float64).reshape(-1, 5, 3)

    # Scale invariance: divide by the mean ear-to-ear distance
    head_size = np.linalg.norm(landmarks[:, 3] - landmarks[:, 4], axis=1).mean()
    landmarks = landmarks.reshape(-1, 15) / max(head_size, 1e-6)

    # Position invariance: keep only the motion around the window mean
    landmarks = landmarks - landmarks.mean(axis=0, keepdims=True)

    return landmarks.reshape(-1).tolist()


class HeadGestureClassifier(object):
    def __init__(
        self,
        model_path='model/head_gesture_classifier/head_gesture_classifier.tflite',
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        head_history,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([head_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index
