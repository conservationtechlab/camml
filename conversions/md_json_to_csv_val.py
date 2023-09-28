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
    parser.add_argument("image_folder_path", type=str,
                        help="path to the image folder")
    parser.add_argument("val_image_path", type=str,
                        help="path to the validation image folder")
    #parser.add_argument("val_label_path", type=str,
    #                    help="path to the validation annotations")
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
            image_path = os.path.join(args.image_folder_path,
                                      img['file'])

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

                    # Randomly set 90% of images to train,
                    # and 10% to test.
                    rand_num = random.randint(1, 100)
                    set_type = ''

                    if rand_num <= 90:
                        set_type = 'TRAIN'
                    elif rand_num <= 100:
                        set_type = 'TEST'

                    # Get the class name from the image file name
                    slash = img['file'].find('/')
                    category = img['file'][0:slash]

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

        # Iterate through OID labels
        # Convert OID annotations to autoML format
        # Write validation annotations to csv
        
        # Go to the label folder path and gather all text files recursively
        txt_files = sorted(glob.glob(args.val_image_path + '/**/*.txt',
                              recursive=True))
        img_files = sorted(glob.glob(args.val_image_path + '/**/*.jpg',
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
                    #print(detections)

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
