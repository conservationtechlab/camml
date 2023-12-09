"""Converts megadetector output JSON file to match CSV format

 CSV format needed for the object_detector.Dataloader.from_csv()
 method as used in this article:
 https://www.tensorflow.org/lite/models/modify/model_maker/object_detection

 Run as:
     python megadetector_json_to_csv.py output.json new_output.csv \
     /home/user/image_folder/ 0.15

 Or to include blank detections(causes an error when training):
     python megadetector_json_to_csv.py output.json new_output.csv \
     /home/user/image_folder/ 0.15 --include
"""

import json
import csv
import argparse
import random

from dataprep import bbox_to_pascal, add_md_detection_to_csv

TESTING = False


def main():
    """Prepares a CSV file for camml TFLite training from MD JSON file

    Converts the JSON xywh bbox data to xyxy format and writes the
    data to a CSV in the proper format [set, class, file, xmin, ymin,
    '', '', xmax, ymax, '', ''].

    """
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

    with open(args.input_file, encoding="utf-8") as json_file:
        data = json.load(json_file)

    image_data = data['images']

    with open(args.output_file, 'w', encoding="utf-8") as data_file:
        csv_writer = csv.writer(data_file)

        for img in image_data:
            # Set the file path
            image_path = img['file']

            # Get the class label from the image file name
            new_label = image_path.strip('/').split('/')[-2]

            if 'failure' in img.keys():
                # NOTE: maybe we should add a more explanatory error
                # message.  Eg what is it that causes the string
                # 'failure' to be in img.keys()?  Is that something
                # comes from MD?
                print(img['file'] + ' failed to access.\n')
            else:
                for detection in img['detections']:
                    # convert bbox to Pascal VOC
                    bbox_to_pascal(detection['bbox'])

                    # Randomly set 80% of images to train, 10% to
                    # validation, and 10% to test.
                    if TESTING:
                        random.seed(42)
                    rand_num = random.randint(1, 100)
                    set_type = ''

                    if rand_num <= 80:
                        set_type = 'TRAIN'
                    elif rand_num <= 90:
                        set_type = 'VALIDATION'
                    elif rand_num <= 100:
                        set_type = 'TEST'

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


if __name__ == "__main__":
    main()
