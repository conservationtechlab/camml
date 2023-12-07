"""Converts megadetector output JSON file to match CSV format

 Separates megadetector output detections into only train and test.
 Validation annotations from OIDv4 boxes.

 CSV format needed for the object_detector.Dataloader.from_csv()
 method as used in this article:
 https://www.tensorflow.org/lite/models/modify/model_maker/object_detection

 Run as:
     python megadetector_json_to_csv.py output.json new_output.csv \
     /home/user/image_folder/ 0.9

 Or to include blank detections(causes an error when training):
     python megadetector_json_to_csv.py output.json new_output.csv \
     /home/user/image_folder/ 0.9 --include
"""

import json
import csv
import os
import argparse
import random
import glob

from PIL import Image

from dataprep import bbox_to_pascal, write_detection_to_csv


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
        # Create the csv writer object
        csv_writer = csv.writer(data_file)

        for img in image_data:
            # Set the file path
            image_path = img['file']
            # Get the class label from the image file name
            new_label = image_path.strip('/').split('/')[-2]

            if 'failure' in img.keys():
                print(img['file'] + ' failed to access.\n')
            else:
                for detection in img['detections']:
                    # convert bbox to Pascal VOC
                    bbox_to_pascal(detection['bbox'])

                    # Randomly set 90% of images to train,
                    # and 10% to test.
                    # random.seed(42)  # for testing
                    rand_num = random.randint(1, 100)
                    set_type = ''

                    if rand_num <= 90:
                        set_type = 'TRAIN'
                    elif rand_num <= 100:
                        set_type = 'TEST'

                    # Get the class name from the image file name

                    write_detection_to_csv(csv_writer, detection,
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

        # Go to the label folder path and gather all text files recursively
        txt_files = sorted(glob.glob(val_img_path + '/**/*.txt',
                                     recursive=True))
        img_files = sorted(glob.glob(val_img_path + '/**/*.jpg',
                                     recursive=True))

        # For each text file change the annotations and write to csv
        for (txt, img) in zip(txt_files, img_files):
            with open(txt,
                      'r', encoding='utf-8') as current_file:
                val_img = Image.open(img)

                width, height = val_img.size

                val_data = current_file.read()

                val_data = val_data.split('\n')
                val_data = val_data[:-1]

                # Set all annotatins to validation
                set_type = 'VALIDATION'

                # For each detection change to normalized xyxy
                for element in val_data:
                    detections = element.split()

                    detections[1] = round(float(detections[1]) / width, 4)
                    detections[2] = round(float(detections[2]) / height, 4)
                    detections[3] = round(float(detections[3]) / width, 4)
                    detections[4] = round(float(detections[4]) / height, 4)

                    csv_writer.writerow([set_type,
                                         img,
                                         detections[0], detections[1],
                                         detections[2], None, None,
                                         detections[3], detections[4],
                                         None, None])


if __name__ == "__main__":
    main()
