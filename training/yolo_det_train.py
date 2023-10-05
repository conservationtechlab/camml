""" Trains a YOLOv8 model with the default parameters.

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
results = model.train(
   data=DATA,
   imgsz=IMAGE_SIZE,
   epochs=EPOCHS,
   batch=BATCH_SIZE,
   name=NAME
)
