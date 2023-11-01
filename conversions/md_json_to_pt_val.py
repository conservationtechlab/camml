"""Converts megadetector output JSON file to match Yolov8 pytortch txt format

A .yaml file with image directory and class information is needed to
train a Yolov8 model. This script will create a "./yolov8_training_data/"
folder containing the symlinked images and produced annotation files.
This script will also create the yaml file pointing to the training and
validation images. The yaml file will have a format similar to
the one described in the "EXAMPLE" section here:
https://roboflow.com/formats/yolov8-pytorch-txt

A confidence value from 0 to 1, such as 0.9, is needed to filter
detections with a confidence below this value.

 Run as:
     python md_json_to_pt_val.py output.json \
     /home/user/validation_images/ 0.9
"""

import json
import argparse
import os
import glob
import yaml

from PIL import Image


def main():
    """Converts JSON data and writes to txt files

    Converts the JSON xywh bbox data to xyxy format and writes the
    data to txt files in the proper format
    (class_id centerx centery width height).
    """
    # pylint: disable=locally-disabled, too-many-locals
    # pylint: disable=locally-disabled, too-many-branches, too-many-statements
    # Get command line arguments when running program
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str,
                        help="filepath for the JSON input file")
    parser.add_argument("val_image_path", type=str,
                        help="filepath for the validation images")
    parser.add_argument("conf", type=float,
                        help="confidence threshold for megadetector \
                        detections")
    args = parser.parse_args()

    # Create the folder directory for our symlinked images and new labels
    home_path = os.path.expanduser('~')
    train_img_path = './yolov8_training_data/train/images/'
    train_label_path = './yolov8_training_data/train/labels/'
    val_img_path = './yolov8_training_data/validation/images/'
    val_label_path = './yolov8_training_data/validation/labels/'

    # Create folder structure for yolo training
    # Add test paths later
    os.makedirs(train_img_path)
    os.makedirs(train_label_path)
    os.makedirs(val_img_path)
    os.makedirs(val_label_path)

    # Opening JSON file and loading the data
    # into the variable
    with open(args.input_file, encoding="utf-8") as json_file:
        data = json.load(json_file)

    image_data = data['images']
    class_list = []

    # For loop get to each image file name
    for img in image_data:
        # Symlink all images to new location
        last_slash = img['file'].rfind('/')
        img_filename = img['file'][last_slash + 1:]
        if os.path.exists(train_img_path + img_filename) is False:
            os.symlink(img['file'], train_img_path + img_filename)

        # First check if file contains detections
        filename = img['file'][last_slash + 1:-3] + 'txt'
        if 'failure' in img.keys():
            print(img['file'] + ' failed to access.\n')
        else:
            # Create a new text file for each image
            with open(train_label_path + filename, 'w+',
                      encoding="utf-8") as text_file:
                for i in range(0, len(img['detections'])):
                    # Megadetector uses 3 categories 1-animal, 2-person,
                    # 3-vehicle, only the animal detections are needed
                    if (img['detections'][i]['category'] == '1'
                            and img['detections'][i]['conf'] >= args.conf):
                        # Convert xmin, ymin, width, height json format
                        # to centerx, centery, width, height yolo format
                        xmin = img['detections'][i]['bbox'][0]
                        ymin = img['detections'][i]['bbox'][1]
                        width = img['detections'][i]['bbox'][2]
                        height = img['detections'][i]['bbox'][3]

                        center_x = xmin + (width / 2)
                        center_y = ymin + (height / 2)

                        center_x = round(center_x, 8)
                        center_y = round(center_y, 8)
                        width = round(width, 8)
                        height = round(height, 8)

                        # Get the class name from the image file name
                        slash = img['file'].rfind('/')
                        prev_slash = img['file'][0:slash - 1].rfind('/')
                        category = img['file'][prev_slash + 1:slash]

                        # Collect all class names and add them to list
                        if category not in class_list:
                            class_list.append(category)

                        # Assign class number to each detection
                        class_num = class_list.index(category)

                        # Separate detections onto different lines
                        text_file.seek(0)
                        first_char = text_file.read(1)
                        if not first_char:
                            line = str(class_num) + ' ' \
                                       + str(center_x) + ' ' \
                                       + str(center_y) + ' ' + str(width) \
                                       + ' ' + str(height)
                            text_file.write(line)
                        else:
                            line = '\n' + str(class_num) + ' ' \
                                       + str(center_x) + ' ' \
                                       + str(center_y) + ' ' + str(width) \
                                       + ' ' + str(height)
                            text_file.write(line)

    # Go to the label folder path and gather all text files recursively
    txt_files = sorted(glob.glob(args.val_image_path + '/**/*.txt',
                                 recursive=True))
    img_files = sorted(glob.glob(args.val_image_path + '/**/*.jpg',
                                 recursive=True))

    # For each text file convert annotations and write to a new text file
    for (txt, img) in zip(txt_files, img_files):
        # Symlink all images to new folder
        last_slash = img.rfind('/')
        if os.path.exists(val_img_path + img[last_slash + 1:]) is False:
            os.symlink(img, val_img_path + img[last_slash + 1:])

        # Read contents of each txt file
        last_slash = txt.rfind('/')
        new_file = txt[last_slash + 1:]
        with open(txt, 'r',
                  encoding='utf-8') as current_file:
            img = Image.open(img)

            img_width, img_height = img.size

            data = current_file.read()

            data = data.split('\n')
            data = data[:-1]

        with open(val_label_path + new_file, 'w+',
                  encoding='utf-8') as text_file:
            for element in data:
                detections = element.split()

                # Convert normalized xyxy to centerx, centery, width, height
                # Need to round to certain amount of digits
                bbox_width = (float(detections[3])
                              - float(detections[1])) / img_width
                bbox_height = (float(detections[4])
                               - float(detections[2])) / img_height

                # Convert xyxy to normalized format
                detections[1] = float(detections[1]) / img_width
                detections[2] = float(detections[2]) / img_height
                detections[3] = float(detections[3]) / img_width
                detections[4] = float(detections[4]) / img_height

                center_x = detections[1] + (bbox_width / 2)
                center_y = detections[2] + (bbox_height / 2)

                # Need to double check all calculations and if rounding is
                # consistent with other formats
                center_x = round(center_x, 8)
                center_y = round(center_y, 8)
                bbox_width = round(bbox_width, 8)
                bbox_height = round(bbox_height, 8)

                # Get class number from text file
                class_num = class_list.index(detections[0])

                # Separate detections onto different lines
                text_file.seek(0)
                first_char = text_file.read(1)
                if not first_char:
                    line = str(class_num) + ' ' + str(center_x) + ' ' \
                           + str(center_y) + ' ' + str(bbox_width) \
                           + ' ' + str(bbox_height)
                    text_file.write(line)
                else:
                    line = '\n' + str(class_num) + ' ' + str(center_x) + ' ' \
                           + str(center_y) + ' ' + str(bbox_width) \
                           + ' ' + str(bbox_height)
                    text_file.write(line)

    # Create new yaml file that contains image directory and class info
    new_yaml = "train: " + home_path + train_img_path[1:] + "\n" \
               + "val: " + home_path + val_img_path[1:] + "\n\n" \
               + "nc: " + str(len(class_list)) + "\n" + "names: " \
               + str(class_list)
    content = yaml.safe_load(new_yaml)

    with open('data.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(content, file, sort_keys=False)


if __name__ == "__main__":
    main()
