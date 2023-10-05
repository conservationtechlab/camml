""" Trains a YOLOv8 model with default parameters.

MODEL can be set to yolov8s.pt, yolov8m.pt, yolov8l.pt, or yolov8x.pt.

DATA is the name of your yaml file containing train, val, and test
image paths as well as class information.

IMAGE_SIZE is set to 640x640, can also be set to 1280x1280.

EPOCHS is set to 50, meaning 50 training epochs.

BATCH_SIZE is set to 8, recommended size is whatever your GPU can handle.

NAME is the name of the training folder where all weights and info will
be saved. After your training there should be a
"./runs/detect/yolov8n_custom/weights/" directory containing the
"best.pt" and "last.pt" trained models. I recommend renaming
this "best.pt" model file to something more specific so you can
test it later.

Run as :
python yolo_det_train.py
"""

from ultralytics import YOLO

MODEL = 'yolov8n.pt'
DATA = 'custom_data.yaml'
IMAGE_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 8
NAME = 'yolov8n_custom'

# Load the model
model = YOLO(MODEL)

# Training
model.train(
   data=DATA,
   imgsz=IMAGE_SIZE,
   epochs=EPOCHS,
   batch=BATCH_SIZE,
   name=NAME
)
