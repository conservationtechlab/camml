"""Prepares training data from MD JSON and OID annotations

Converts megadetector output JSON file and OID annotationsto match CSV
format

Training annotations come from detections from MegaDetector
 
Validation and test annotations come from OIDv4 boxes.

There are some assumptions about layout of data that should be
documented.

CSV format needed for the object_detector.Dataloader.from_csv() method
as used in this article:
https://www.tensorflow.org/lite/models/modify/model_maker/object_detection

"""

import json
import csv
import os
import argparse
import glob

from PIL import Image

from dataprep import bbox_to_pascal
from dataprep import add_md_detection_to_csv, add_oid_annotations_to_csv


def main():
    """Converts JSON data and writes to CSV

    Converts the JSON xywh bbox data to xyxy format and writes the
    data to a CSV in the proper format [set, class, file, xmin, ymin,
    '', '', xmax, ymax, '', ''].
    """
    # pylint: disable=locally-disabled, too-many-locals, too-many-statements
    # Get command line arguments when running program
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str,
                        help="filepath for the JSON input file")
    parser.add_argument("output_file", type=str,
                        help="filepath for the CSV output file")
    parser.add_argument("conf", type=float,
                        help="confidence threshold")
    parser.add_argument("--include",
                        help="include blank detections in csv",
                        dest='include', action='store_true')
    args = parser.parse_args()

    # Opening JSON file and loading the data
    # into the variable data
    with open(args.input_file, encoding="utf-8") as json_file:
        data = json.load(json_file)

    image_data = data['images']

    # Open a file for writing
    with open(args.output_file, 'w', encoding="utf-8") as data_file:
        csv_writer = csv.writer(data_file)

        for img in image_data:
            image_path = img['file']
            new_label = image_path.strip('/').split('/')[-2]
            set_type = 'TRAIN'

            if 'failure' in img.keys():
                print(img['file'] + ' failed to access.\n')
            else:
                for detection in img['detections']:
                    bbox_to_pascal(detection['bbox'])
                    add_md_detection_to_csv(csv_writer, detection,
                                            set_type, image_path,
                                            new_label,
                                            confidence_threshold=args.conf)

                # To make images with no detections appear in the csv file
                if args.include:
                    if len(img['detections']) == 0:
                        csv_writer.writerow([set_type,
                                             image_path,
                                             None, None, None,
                                             None, None, None,
                                             None, None, None])

        # Iterate through OID labels
        # Convert OID annotations to autoML format
        # Write validation annotations to csv
        folder = image_path.rfind('train')
        val_img_path = os.path.join(image_path[:folder] + 'validation/')
        test_img_path = os.path.join(image_path[:folder] + 'test/')

        # Go to the label folder path and gather all text files recursively
        val_txt_files = sorted(glob.glob(val_img_path + '/**/*.txt',
                                         recursive=True))
        val_img_files = sorted(glob.glob(val_img_path + '/**/*.jpg',
                                         recursive=True))

        test_txt_files = sorted(glob.glob(test_img_path + '/**/*.txt',
                                          recursive=True))
        test_img_files = sorted(glob.glob(test_img_path + '/**/*.jpg',
                                          recursive=True))

        for (val_txt, val_img) in zip(val_txt_files, val_img_files):
            add_oid_annotations_to_csv(csv_writer, val_img, val_txt, 'VALIDATION')

        for (test_txt, test_img) in zip(test_txt_files, test_img_files):
            add_oid_annotations_to_csv(csv_writer, test_img, test_txt, 'TEST')


if __name__ == "__main__":
    main()
