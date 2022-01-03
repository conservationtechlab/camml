"""Test coral object detection on frames from webcam

Captures from webcam (assumes attached webcam) continuously and runs
object detector on Coral USB accelerator (also assumes is attached) on
captured frames, reporting detected objects to terminal.

"""
import os
import sys

import cv2

from camml.coral import ObjectDetectorHandler

MODEL_PATH = '/home/ian/git/coral/pycoral/test_data/'
MODEL_CONFIG_FILE = 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
CLASS_NAMES_FILE = "coco_labels.txt"

INPUT_WIDTH = INPUT_HEIGHT = 416
SCORE_THRESHOLD = .7
NMS_THRESHOLD = .4

MODEL_CONFIG = os.path.join(MODEL_PATH,
                            MODEL_CONFIG_FILE)
CLASSES_FILE = os.path.join(MODEL_PATH,
                            CLASS_NAMES_FILE)
CLASSES = []
for row in open(CLASSES_FILE):
    CLASSES.append(row.strip())

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
            print("----{:.2f} milliseconds".format(inf_time))

            for lbox in lboxes:
                label = CLASSES[lbox['class_id']]
                score = lbox['confidence']
                print("{} | {:.2f}".format(label, score))
except KeyboardInterrupt:
    print('Received keyboard interrupt.')
    sys.exit()
