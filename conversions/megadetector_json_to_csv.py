"""Converts megadetector output JSON file to match CSV format

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
import argparse
import random

from dataprep import bbox_to_pascal


def main():
    """Prepares a CSV file for camml TFLite training from MD JSON file

    Converts the JSON xywh bbox data to xyxy format and writes the
    data to a CSV in the proper format [set, class, file, xmin, ymin,
    '', '', xmax, ymax, '', ''].

    """
    # pylint: disable=locally-disabled, too-many-locals
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
            new_label = img['file'].strip('/').split('/')[-2]
            
            if 'failure' in img.keys():
                print(img['file'] + ' failed to access.\n')
            else:
                for detection in img['detections']:
                    # convert bbox to Pascal VOC
                    bbox_to_pascal(detection['bbox'])
                    
                    # Randomly set 80% of images to train,
                    # 10% to validation, and 10% to test. Currently all set
                    # random.seed(42)  # for testing
                    rand_num = random.randint(1, 100)
                    set_type = ''

                    if rand_num <= 80:
                        set_type = 'TRAIN'
                    elif rand_num <= 90:
                        set_type = 'VALIDATION'
                    elif rand_num <= 100:
                        set_type = 'TEST'

                    # Megadetector uses 3 categories 1-animal, 2-person,
                    # 3-vehicle, only the animal detections are needed
                    # Filters detections so only >= conf detections appear
                    if (detection['category'] == '1'
                            and detection['conf'] >= args.conf):
                        csv_writer.writerow([set_type,
                                             image_path,
                                             new_label,
                                             detection['bbox'][0],
                                             detection['bbox'][1],
                                             None, None,
                                             detection['bbox'][2],
                                             detection['bbox'][3],
                                             None, None])

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
