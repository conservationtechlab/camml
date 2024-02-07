"""Test coral object detection on frames from webcam

Captures from webcam (assumes attached webcam) continuously and runs
object detector on Coral USB accelerator (also assumes is attached) on
captured frames, reporting detected objects to terminal.

Example usage:

python test_detection_with_webcam.py --threshold .2 \
--model ~/tflite_train/eleven_animals/eleven_animals_e25_edgetpu.tflite \
--labelmap ~/tflite_train/eleven_animals/labelmap.txt

"""
import sys
import argparse

import cv2

from camml.coral import ObjectDetectorHandler

parser = argparse.ArgumentParser()
parser.add_argument('-t',
                    '--threshold',
                    default=.1
                    )
parser.add_argument('-l',
                    '--labelmap',
                    default=.1
                    )
parser.add_argument('-m',
                    '--model',
                    default=.1
                    )

args = parser.parse_args()

MODEL_CONFIG = args.model
CLASSES_FILE = args.labelmap

SCORE_THRESHOLD = float(args.threshold)
INPUT_WIDTH = INPUT_HEIGHT = 416
NMS_THRESHOLD = .4

with open(CLASSES_FILE, 'r', encoding='utf-8') as file:
    CLASSES = file.read().splitlines()

print("Using these classes: " + ', '.join(CLASSES))

detector = ObjectDetectorHandler(MODEL_CONFIG,
                                 None,
                                 INPUT_WIDTH,
                                 INPUT_HEIGHT)

cap = cv2.VideoCapture(0)
ok, frame = cap.read()

try:
    while True:
        ok, frame = cap.read()
        objs, inf_time = detector.infer(frame)

        lboxes = detector.filter_boxes(objs,
                                       frame,
                                       SCORE_THRESHOLD,
                                       .2)

        if lboxes:
            print(f"----{inf_time:.2f} milliseconds")

            for lbox in lboxes:
                label = CLASSES[lbox['class_id']]
                score = lbox['confidence']
                print(f"{label} | {score:.2f}")
except KeyboardInterrupt:
    print('Received keyboard interrupt.')
    sys.exit()
