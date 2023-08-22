"""Converts megadetector output JSON file to match Yolov8 pytortch txt format

 CSV format needed for the object_detector.Dataloader.from_csv()
 method as used in this article:
 https://www.tensorflow.org/lite/models/modify/model_maker/object_detection

 Run as:
     python megadetector_json_to_csv.py output.json new_output.csv \
     /home/user/image_folder/

 Or to include blank detections(causes an error when training):
     python megadetector_json_to_csv.py output.json new_output.csv \
     /home/user/image_folder/ --include
"""

import json
import csv
import os
import argparse
import random
import yaml
import shutil


def main():
    """Converts JSON data and writes to txt files

    Converts the JSON xywh bbox data to xyxy format and writes the
    data to txt files in the proper format [set, class, file, xmin, ymin,
    '', '', xmax, ymax, '', ''].
    """
    # pylint: disable=locally-disabled, too-many-locals
    # Get command line arguments when running program
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str,
                        help="filepath for the JSON input file")
    parser.add_argument("output_folder", type=str,
                        help="filepath for the txt output files")
    #parser.add_argument("image_folder_path", type=str,
    #                    help="path to the image folder")
    parser.add_argument("confidence", type=float,
                        help="confidence threshold for megadetector detections")
    args = parser.parse_args()

    # Opening JSON file and loading the data
    # into the variable data
    with open(args.input_file, encoding="utf-8") as json_file:
        data = json.load(json_file)

    image_data = data['images']



    # For loop get to each image file name
    for img in image_data:
        
        # First check if file contains detections
        filename = img['file'][:-3] + 'txt'
        if 'failure' in img.keys():
            print(img['file'] + ' failed to access.\n')
        else:
            # Create a new text file for each image
            with open(args.output_folder + filename, 'w+') as text_file:
                for i in range(0, len(img['detections'])):
                    # Megadetector uses 3 categories 1-animal, 2-person,
                    # 3-vehicle, only the animal detections are needed
                    if img['detections'][i]['category'] == '1' and img['detections'][i]['conf'] >= args.confidence:
                        # Convert xmin, ymin, width, height json format to centerx, centery, width, height yolo format
                        xmin = img['detections'][i]['bbox'][0]
                        ymin = img['detections'][i]['bbox'][1]
                        width = img['detections'][i]['bbox'][2]
                        height = img['detections'][i]['bbox'][3]

                        center_x = xmin + (width / 2)
                        center_y = ymin + (height / 2)
                        
                        # Separate detections onto different lines
                        # Dont create text file if no detections
                        text_file.seek(0)
                        first_char = text_file.read(1)
                        if not first_char:
                            line = '0 ' + str(center_x) + ' ' + str(center_y) + ' ' + str(width) + ' ' + str(height)
                            text_file.write(line)
                        else:
                            line = '\n0 ' + str(center_x) + ' ' + str(center_y) + ' ' + str(width) + ' ' + str(height)
                            text_file.write(line)


    # Create new yaml file that contains image directory and class info
    new_yaml = """
train: /home/ericescareno/OIDv4_ToolKit/OID/Dataset/train/images
val: /home/ericescareno/OIDv4_ToolKit/OID/Dataset/validation/images  
  
nc: 1
names: ['Cat']
"""
    content = yaml.safe_load(new_yaml)

    with open('data.yaml', 'w') as file:
        yaml.dump(content, file, sort_keys=False)




'''
    # Delete the empty .txt files with no detections that fit our criteria
    # Move the images corresponding to these text files to a different folder
    no_of_files_deleted = 0
    
    # Route through the directories and files within the path -
    for (dir, _, files) in os.walk(args.output_folder):
        for filename in files:
            img_filename = filename[:-3] + 'jpg'
            # Generate file path
            file_path = os.path.join(dir, filename)
            image_path = os.path.join(args.image_folder_path, img_filename)
  
            # Check if it is file and empty (size = 0)
            if (
                os.path.isfile(file_path) and
                os.path.getsize(file_path) == 0
            ):
  
                # Print the path of the file that will be deleted
                print("Deleting File >>>", file_path.replace('\\', '/'))
                print("Moving File >>>", image_path.replace('\\', '/'))
  
                # Delete the empty file 
                os.remove(file_path)
                #os.remove(image_path)
                shutil.move(image_path, "/home/ericescareno/coral/pycoraltest/openimages_test/OIDv4_ToolKit/OID/Dataset/train/blank_detections/" + img_filename)
  
                #no_of_files_deleted += 2
  
    #print(no_of_files_deleted, "file(s) have been deleted.")
'''





if __name__ == "__main__":
    main()
