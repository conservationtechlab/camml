"""Test coral object detection on frames from picamera

Captures from picamera (assumes attached picamera) continuously and
runs object detector on Coral USB accelerator (also assumes is
attached) on captured frames, reporting detected objects to terminal.

Example usage:

python test_detection_with_picamera.py --threshold .2 \
--model ~/tflite_train/eleven_animals/eleven_animals_e25_edgetpu.tflite \
--labelmap ~/tflite_train/eleven_animals/labelmap.txt

"""
import sys
import argparse

from picamera2 import Picamera2

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

picam2 = Picamera2()
config = picam2.create_preview_configuration({"format": 'BGR888'})
picam2.configure(config)
picam2.start()

try:
    while True:
        # WATCH OUT: will there be an unaddressed RGB/BGR shift here?
        # detector.infer definitely assumes BGR Should be addressed in
        # the picamera config above but needs to be double-checked
        frame = picam2.capture_array()

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