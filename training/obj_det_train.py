"""Retrain object detection model and export to tflite model

Retrains a TFLite *EfficientDet Lite0* object detection model from
custom data with custom classes.

"""
import os
import argparse
import tensorflow as tf
from absl import logging

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

# Set object detection model architecture
spec = model_spec.get('efficientdet_lite0')

# Get input csv file as command line argument
parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str,
                    help="Filepath for the CSV input file")
parser.add_argument("output_model_path", type=str,
                    help="Filepath for created model file.")
parser.add_argument('-m',
                    "--model_name",
                    default='model.tflite',
                    type=str,
                    help="name")
parser.add_argument("-e",
                    "--num_epochs",
                    type=int,
                    default=1,
                    help="Number of epochs for training")

args = parser.parse_args()

# Load dataset
# pylint: disable=unbalanced-tuple-unpacking
train_data, validation_data, test_data = \
    object_detector.DataLoader.from_csv(args.input_file)

# Train TF model with training data
model = object_detector.create(train_data, model_spec=spec,
                               batch_size=8, epochs=args.num_epochs,
                               train_whole_model=True,
                               validation_data=validation_data)

print('\n*** Exporting TFLite model ***\n')
model.export(export_dir=args.output_model_path,
             tflite_filename=args.model_name)

print('\n*** Running evaluation of model with test data ***\n')
print(model.evaluate(test_data))

print('\n*** Running evaluation of TFLITE model with test data ***\n')
model_path = os.path.join(args.output_model_path, args.model_name)
print(model.evaluate_tflite(model_path, test_data))
