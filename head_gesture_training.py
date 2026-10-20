#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the head gesture classifier on the CCDb-HG extracted features.

Only the 5 averaged Face Mesh landmarks are used (not the dataset's head pose), so live
inference only needs mp.solutions.face_mesh. Adding head pose improved test macro F1 by ~0.01.

Dataset: CCDb-HG (Idiap Research Institute), CC BY-NC-SA 4.0
  https://github.com/idiap/ccdbhg-head-gesture-recognition
  file: data/extraction/samples_video_hp_mediapipe_v1_g_xgaze_with_FaceAlignPnPHeadPose_lm_mediapipe_v1_selected.p

Usage:
  python head_gesture_training.py --data path/to/ccdbhg_features.p
"""
import argparse
import json
import pickle

import numpy as np
import tensorflow as tf

from model.head_gesture_classifier.head_gesture_classifier import (
    WINDOW_SIZE, pre_process_head_history)

RANDOM_SEED = 42
LABELS = ['None', 'Nod', 'Shake', 'Tilt', 'Turn', 'Up/Down']
NUM_CLASSES = len(LABELS)
NUM_CHANNELS = 15

# Subject-independent split (test subjects match the CCDb-HG authors' test set)
TEST_SUBJECTS = {'P2', 'P5', 'P10', 'P20'}
VAL_SUBJECTS = {'P3', 'P15'}

MODEL_DIR = 'model/head_gesture_classifier/'


class SafeUnpickler(pickle.Unpickler):
    # The features file only needs numpy scalars/dtypes; refuse anything else
    def find_class(self, module, name):
        if (module, name) in {('numpy.core.multiarray', 'scalar'), ('numpy', 'dtype')}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'Blocked global {module}.{name}')


def load_videos(path):
    with open(path, 'rb') as f:
        return SafeUnpickler(f).load()


def mirror(landmarks):
    # Horizontal flip: negate x, swap left/right eye and ear
    lm = landmarks.reshape(-1, 5, 3).copy()
    lm[:, :, 0] = -lm[:, :, 0]
    lm = lm[:, [1, 0, 2, 4, 3], :]
    return lm.reshape(-1, 15)


def build_samples(videos, gesture_stride, none_stride, augment):
    x, y = [], []
    half = WINDOW_SIZE // 2
    for video in videos:
        landmarks = np.asarray(video['landmarks'], dtype=np.float64)
        errors = np.asarray(video['errors_extraction']) != 0
        labels = np.array([label[0] for label in video['label']])
        labels[labels == 6] = 0  # Waggle is too rare; treated as None (as the authors do)

        for center in range(half, len(labels) - half):
            label = labels[center]
            if center % (gesture_stride if label != 0 else none_stride):
                continue
            window = slice(center - half, center + half + 1)
            if errors[window].any():
                continue
            x.append(pre_process_head_history(landmarks[window]))
            y.append(label)
            if augment:
                x.append(pre_process_head_history(mirror(landmarks[window])))
                y.append(label)
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int64)


def build_model(x_train):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(x_train.reshape(-1, NUM_CHANNELS))

    return tf.keras.models.Sequential([
        tf.keras.layers.Input((WINDOW_SIZE * NUM_CHANNELS, )),
        tf.keras.layers.Reshape((WINDOW_SIZE, NUM_CHANNELS)),
        normalizer,
        tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])


def evaluate(y_true, y_pred):
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    per_class = {}
    for i, name in enumerate(LABELS):
        tp = confusion[i, i]
        precision = tp / max(confusion[:, i].sum(), 1)
        recall = tp / max(confusion[i, :].sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[name] = {'precision': round(float(precision), 3), 'recall': round(float(recall), 3),
                           'f1': round(float(f1), 3), 'support': int(confusion[i, :].sum())}
    return {
        'accuracy': round(float(np.trace(confusion) / confusion.sum()), 3),
        'macro_f1': round(float(np.mean([c['f1'] for c in per_class.values()])), 3),
        'macro_f1_gestures': round(float(np.mean([per_class[n]['f1'] for n in LABELS[1:]])), 3),
        'per_class': per_class,
        'confusion_matrix': confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to the CCDb-HG features pickle')
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    videos = load_videos(args.data)
    train_videos = [v for v in videos if v['info']['subject'] not in TEST_SUBJECTS | VAL_SUBJECTS]
    val_videos = [v for v in videos if v['info']['subject'] in VAL_SUBJECTS]
    test_videos = [v for v in videos if v['info']['subject'] in TEST_SUBJECTS]

    # Training: gestures densely, None sparsely, plus mirrored copies
    x_train, y_train = build_samples(train_videos, gesture_stride=2, none_stride=12, augment=True)
    # Validation/test: uniform stride keeps the natural class distribution
    x_val, y_val = build_samples(val_videos, gesture_stride=5, none_stride=5, augment=False)
    x_test, y_test = build_samples(test_videos, gesture_stride=5, none_stride=5, augment=False)
    for name, y in (('train', y_train), ('val', y_val), ('test', y_test)):
        print(name, len(y), np.bincount(y, minlength=NUM_CLASSES).tolist())

    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weight = {i: float(np.sqrt(counts.max() / counts[i])) for i in range(NUM_CLASSES)}

    model = build_model(x_train)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        ],
        verbose=2,
    )

    metrics = {
        'val': evaluate(y_val, model.predict(x_val, verbose=0).argmax(axis=1)),
        'test': evaluate(y_test, model.predict(x_test, verbose=0).argmax(axis=1)),
        'test_subjects': sorted(TEST_SUBJECTS),
        'num_samples': {'train': len(y_train), 'val': len(y_val), 'test': len(y_test)},
    }
    print(json.dumps({k: metrics[k] for k in ('val', 'test')}, indent=1))

    model.save(MODEL_DIR + 'head_gesture_classifier.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(MODEL_DIR + 'head_gesture_classifier.tflite', 'wb') as f:
        f.write(converter.convert())
    with open(MODEL_DIR + 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=1)


if __name__ == '__main__':
    main()
