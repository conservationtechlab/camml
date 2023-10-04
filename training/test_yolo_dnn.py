"""Converts YOLOv8 .pt model file to .onnx model file and tests inference

This script takes 3 command line arguments model_file, image_file,
and yaml_file. The model_file argument should be given the path
to the .pt model file you've trained. The image_file argument
should be given the path to the image file you plan to test inference
on. The yaml_file argument should be given the path to the data.yaml
file you used to train your YOLOv8 model with, so it can extract the
class names.

Run as:
    python test_yolo_dnn.py /home/usr/yolo_model.pt \
    /home/usr/data.yaml /home/usr/image_file.jpg
"""

import argparse
import os
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import yaml

# pylint: disable=I1101, E1101
# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("model_file", type=str,
                help="file path for the .pt model file")
ap.add_argument("yaml_file", type=str,
                help="file path for the data.yaml file")
ap.add_argument("image_file", type=str,
                help="file path for the test image")
args = ap.parse_args()

model = YOLO(args.model_file)
model_name = args.model_file[:-2]
model_onnx = model_name + 'onnx'

# If onnx model file does not exist create it
if not os.path.exists(model_onnx):
    # Export the model to ONNX format
    model.export(format="onnx", imgsz=[640, 640], opset=12)

# Replace onnx filename with model_file + .onnx
net = cv2.dnn.readNet(model_onnx)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

# Define yolov8 classes
with open(args.yaml_file, 'r', encoding="utf-8") as file:
    data = yaml.safe_load(file)
CLASSES_YOLO = data['names']

period = args.image_file.rfind('.')
image_type = args.image_file[period:]

image = cv2.imread(args.image_file)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT),
                             swapRB=True, crop=False)
net.setInput(blob)
preds = net.forward()
preds = preds.transpose((0, 2, 1))

# Extract output detection
class_ids, confs, boxes = [], [], []

image_height, image_width, _ = image.shape
x_factor = image_width / INPUT_WIDTH
y_factor = image_height / INPUT_HEIGHT

rows = preds[0].shape[0]

for i in range(rows):
    row = preds[0][i]
    conf = row[4]

    classes_score = row[4:]
    _, _, _, max_idx = cv2.minMaxLoc(classes_score)
    class_id = max_idx[1]
    if classes_score[class_id] > .25:
        confs.append(conf)
        label = CLASSES_YOLO[int(class_id)]
        class_ids.append(label)

        # Extract boxes
        x, y, w, h = row[0].item(), row[1].item(), \
            row[2].item(), row[3].item()
        left = int((x - 0.5 * w) * x_factor)
        top = int((y - 0.5 * h) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        box = np.array([left, top, width, height])
        boxes.append(box)

r_class_ids, r_confs, r_boxes = [], [], []

indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45)
for i in indexes:
    r_class_ids.append(class_ids[i])
    r_confs.append(confs[i])
    r_boxes.append(boxes[i])

for i in indexes:
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]

    cv2.rectangle(image, (left, top), (left + width, top + height),
                  (0, 255, 0), 3)
img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
img.show()

# Save result image
boxed_image = args.image_file[:period] + '_results' + image_type
img.save(boxed_image)
