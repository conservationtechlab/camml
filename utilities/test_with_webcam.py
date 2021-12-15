"""Test coral classification on frames from webcam

"""
import os
import sys

import cv2

from camml.coral import ImageClassifierHandler

TOP_K = 1

MODEL_PATH = '/home/ian/git/coral/pycoral/test_data/'
MODEL_CONFIG_FILE = 'mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite'
CLASS_NAMES_FILE = "inat_bird_labels.txt"

model = os.path.join(MODEL_PATH,
                     MODEL_CONFIG_FILE)
CLASSES_FILE = os.path.join(MODEL_PATH,
                            CLASS_NAMES_FILE)
CLASSES = []
for row in open(CLASSES_FILE):
    CLASSES.append(row.strip())

classifier = ImageClassifierHandler(model,
                                    top_k=TOP_K)

cap = cv2.VideoCapture(0)
ok, frame = cap.read()

try:
    while True:
        ok, frame = cap.read()
        results, inf_time = classifier.infer(frame)

        if results:
            print("----{:.2f} milliseconds".format(inf_time))

            for result in results:
                label = CLASSES[result.id]
                score = result.score
                print("{} | {:.2f}".format(label, score))
except KeyboardInterrupt:
    print('Received keyboard interrupt.')
    sys.exit()
