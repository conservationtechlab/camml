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


def main():
    """Converts JSON data and writes to CSV

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

            if 'failure' in img.keys():
                print(img['file'] + ' failed to access.\n')
            else:
                for i in range(0, len(img['detections'])):
                    # Convert xywh bbox to xmin, ymin, xmax, ymax bbox
                    x_min, y_min, x_max, y_max = coco_to_pascal_voc(
                        img['detections'][i]['bbox'][0],
                        img['detections'][i]['bbox'][1],
                        img['detections'][i]['bbox'][2],
                        img['detections'][i]['bbox'][3])

                    # Round to 4 digits to match CSV format
                    x_min, y_min = round(x_min, 4), round(y_min, 4)
                    x_max, y_max = round(x_max, 4), round(y_max, 4)

                    # Set bbox coordinates to new values
                    img['detections'][i]['bbox'][0] = x_min
                    img['detections'][i]['bbox'][1] = y_min
                    img['detections'][i]['bbox'][2] = x_max
                    img['detections'][i]['bbox'][3] = y_max

                    # Randomly set 80% of images to train,
                    # 10% to validation, and 10% to test. Currently all set
                    rand_num = random.randint(1, 100)
                    set_type = ''

                    if rand_num <= 80:
                        set_type = 'TRAIN'
                    elif rand_num <= 90:
                        set_type = 'VALIDATION'
                    elif rand_num <= 100:
                        set_type = 'TEST'

                    # Get the class name from the image file name
                    slash = img['file'].rfind('/')
                    prev_slash = img['file'][0:slash - 1].rfind('/')
                    category = img['file'][prev_slash + 1:slash]

                    # Megadetector uses 3 categories 1-animal, 2-person,
                    # 3-vehicle, only the animal detections are needed
                    # Filters detections so only >= conf detections appear
                    if (img['detections'][i]['category'] == '1'
                            and img['detections'][i]['conf'] >= args.conf):
                        csv_writer.writerow([set_type,
                                             image_path,
                                             category,
                                             img['detections'][i]['bbox'][0],
                                             img['detections'][i]['bbox'][1],
                                             None, None,
                                             img['detections'][i]['bbox'][2],
                                             img['detections'][i]['bbox'][3],
                                             None, None])

                # To make images with no detections appear in the csv file
                if args.include:
                    if len(img['detections']) == 0:
                        csv_writer.writerow([set_type,
                                             image_path,
                                             None, None, None,
                                             None, None, None,
                                             None, None, None])


def coco_to_pascal_voc(x_tl, y_tl, width, height):
    """Convert Coco bounding box to Pascal Voc bounding box

    Parameters:
        x_tl (float):Float representing the min x coordinate of the bbox
        y_tl (float):Float representing the min y coordinate of the bbox
        width (float):Float representing the width of the bbox
        height (float):Float representing the height of the bbox

    Returns:
        x_min (float):Float representing the min x coordinate of the bbox
        y_min (float):Float representing the min y coordinate of the bbox
        x_max (float):Float representing the max x coordinate of the bbox
        y_max (float):Float representing the max y coordinate of the bbox
    """
    return [x_tl, y_tl, x_tl + width, y_tl + height]


if __name__ == "__main__":
    main()
