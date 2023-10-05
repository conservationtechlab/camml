"""Converts megadetector output JSON file to match Yolov8 pytortch txt format

A .yaml file with image directory and class information is needed to
train a Yolov8 model. The yaml file will need a format similar to
the one described in the "EXAMPLE" section here:
https://roboflow.com/formats/yolov8-pytorch-txt

A confidence value from 0 to 1, such as 0.9, is needed to filter
detections with a confidence below this value.

 Run as:
     python megadetector_json_to_pt.py output.json \
     /home/user/output_folder/ 0.9
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
    # Get command line arguments when running program
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str,
                        help="filepath for the JSON input file")
    #parser.add_argument("output_folder", type=str,
    #                    help="filepath for the txt output files")
    #parser.add_argument("val_image_path", type=str,
    #                    help="filepath for the validation images")
    #parser.add_argument("val_label_path", type=str,
    #                    help="path to the validation labels")
    #parser.add_argument("val_output_folder", type=str,
    #                    help="filepath for the val txt output files")
    parser.add_argument("conf", type=float,
                        help="confidence threshold for megadetector \
                        detections")
    args = parser.parse_args()

    # Create the folder directory for our symlinked images and new labels
    home_path = os.path.expanduser('~')
    train_image_path = './yolov8_training_data/train/images/'
    train_label_path = './yolov8_training_data/train/labels/'
    val_image_path = './yolov8_training_data/validation/images/'
    val_label_path = './yolov8_training_data/validation/labels/'
    test_image_path = './yolov8_training_data/test/images/'
    test_label_path= './yolov8_training_data/test/labels/'

    # Create folder structure for yolo training
    # Add test paths later
    os.makedirs(train_image_path)
    os.makedirs(train_label_path)
    os.makedirs(val_image_path)
    os.makedirs(val_label_path)
    os.makedirs(test_image_path)
    os.makedirs(test_label_path)

    # Opening JSON file and loading the data
    # into the variable
    with open(args.input_file, encoding="utf-8") as json_file:
        data = json.load(json_file)

    image_data = data['images']
    class_list = []

    # For loop get to each image file name
    for img in image_data:
        # Get image folder path
        end = img['file'].rfind('/')
        prev = img['file'][:end].rfind('/')
        image_folder = img['file'][:prev]

        image_path = img['file']

        # Symlink all images to new location
        last_slash = img['file'].rfind('/')
        img_filename = img['file'][last_slash + 1:]
        if os.path.exists(train_image_path + img_filename) == False:
            os.symlink(img['file'], train_image_path + img_filename)
        
        # First check if file contains detections
        #last_slash = img['file'].rfind('/')
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
                        category = img['file'].strip('/').split('/')[-2]

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
                        
    # Go to the label folder path and gather all text files
    #all_files = os.listdir(args.val_label_path)
    #txt_files = list(filter(lambda x: x[-4:] == '.txt', all_files))

    # For each text file convert annotations and write to a new text file
    for (val_txt, val_img) in zip(val_txt_files, val_img_files):
        # Symlink all images to new folder
        last_slash = val_img.rfind('/')
        if os.path.exists(val_image_path + val_img[last_slash + 1:]) == False:
            os.symlink(val_img, val_image_path + val_img[last_slash + 1:])

        # Read contents of each txt file
        last_slash = val_txt.rfind('/')
        new_file = val_txt[last_slash + 1:]
        with open(val_txt, 'r',
                  encoding='utf-8') as current_file:
            #img_name = txt[:-4] + '.jpg'
            #img = Image.open(args.val_image_path + img_name)
            img = Image.open(val_img)

            img_width, img_height = img.size

            val_data = current_file.read()

            val_data = val_data.split('\n')
            val_data = val_data[:-1]

        with open(val_label_path + new_file, 'w+',
                  encoding='utf-8') as text_file:
            for element in val_data:
                detections = element.split()

                # Convert xyxy to normalized format
                #detections[1] = round(float(detections[1]) / img_width, 4)
                #detections[2] = round(float(detections[2]) / img_height, 4)
                #detections[3] = round(float(detections[3]) / img_width, 4)
                #detections[4] = round(float(detections[4]) / img_height, 4)

                # Convert normalized xyxy to centerx, centery, width, height
                # Need to round to certain amount of digits
                bbox_width = (float(detections[3]) - float(detections[1])) / img_width
                bbox_height = (float(detections[4]) - float(detections[2])) / img_height
                
                # Convert xyxy to normalized format
                detections[1] = float(detections[1]) / img_width
                detections[2] = float(detections[2]) / img_height
                detections[3] = float(detections[3]) / img_width
                detections[4] = float(detections[4]) / img_height
                
                #center_x = (float(detections[1]) / img_width) + (bbox_width / 2)
                #center_y = (float(detections[2]) / img_height) + (bbox_height / 2)

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





    # For each text file convert annotations and write to a new text file
    for (test_txt, test_img) in zip(test_txt_files, test_img_files):
        # Symlink all images to new folder
        last_slash = test_img.rfind('/')
        if os.path.exists(test_image_path + test_img[last_slash + 1:]) == False:
            os.symlink(test_img, test_image_path + test_img[last_slash + 1:])

        # Read contents of each txt file
        last_slash = test_txt.rfind('/')
        new_file = test_txt[last_slash + 1:]
        with open(test_txt, 'r',
                  encoding='utf-8') as current_file:
            #img_name = txt[:-4] + '.jpg'
            #img = Image.open(args.val_image_path + img_name)
            img = Image.open(test_img)

            img_width, img_height = img.size

            test_data = current_file.read()

            test_data = test_data.split('\n')
            test_data = test_data[:-1]

        with open(test_label_path + new_file, 'w+',
                  encoding='utf-8') as text_file:
            for element in test_data:
                detections = element.split()

                # Convert xyxy to normalized format
                #detections[1] = round(float(detections[1]) / img_width, 4)
                #detections[2] = round(float(detections[2]) / img_height, 4)
                #detections[3] = round(float(detections[3]) / img_width, 4)
                #detections[4] = round(float(detections[4]) / img_height, 4)

                # Convert normalized xyxy to centerx, centery, width, height
                # Need to round to certain amount of digits
                bbox_width = (float(detections[3]) - float(detections[1])) / img_width
                bbox_height = (float(detections[4]) - float(detections[2])) / img_height
                
                # Convert xyxy to normalized format
                detections[1] = float(detections[1]) / img_width
                detections[2] = float(detections[2]) / img_height
                detections[3] = float(detections[3]) / img_width
                detections[4] = float(detections[4]) / img_height
                
                #center_x = (float(detections[1]) / img_width) + (bbox_width / 2)
                #center_y = (float(detections[2]) / img_height) + (bbox_height / 2)

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
    new_yaml = "train: " + home_path + train_image_path[1:] + "\n" + "val: " + home_path + val_image_path[1:] + "\n" + "test: " +  home_path + test_image_path[1:] + "\n\n" + "nc: " + str(len(class_list)) + "\n" + "names: " + str(class_list)  
    content = yaml.safe_load(new_yaml)

    with open('data.yaml', 'w') as file:
        yaml.dump(content, file, sort_keys=False)



if __name__ == "__main__":
    main()
