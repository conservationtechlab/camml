"""Retrain object detection model and export to tflite model
"""
import numpy as np
import os
import argparse

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Set object detection model architecture
spec = model_spec.get('efficientdet_lite0')

# Get input csv file as command line argument
parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str, help="filepath for the CSV input file")
args = parser.parse_args()

# Load dataset
train_data, validation_data, test_data = object_detector.DataLoader.from_csv(args.input_file)

# Train TF model with training data
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)

# Evaluate model with test data
model.evaluate(test_data)

# Export as tflite model
model.export(export_dir='.')

# Evaluate tflite model
model.evaluate_tflite('model.tflite', test_data)
