"""Converts OIDv4 annotations to match CSV format

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

import csv
import os
import PIL
import argparse
import random

from PIL import Image

def main():
    """Converts OIDv4 annotation data and writes to CSV

    Converts the OIDv4 denormalized xyxy bbox data
    to normalized xyxy format and writes the
    data to a CSV in the proper format [set, class, file, xmin, ymin,
    '', '', xmax, ymax, '', ''].
    """
    # pylint: disable=locally-disabled, too-many-locals
    # Get command line arguments when running program
    parser = argparse.ArgumentParser()
    parser.add_argument("label_folder_path", type=str,
                        help="path for the folder containing txt input files")
    parser.add_argument("output_file", type=str,
                        help="filepath for the CSV output file")
    parser.add_argument("image_folder_path", type=str,
                        help="path to the image folder")
    #parser.add_argument("--include",
    #                    help="include blank detections in csv",
    #                    dest='include', action='store_true')
    args = parser.parse_args()

    with open(args.output_file, 'w', encoding='utf-8') as data_file:
        csv_writer = csv.writer(data_file)


        # Go to the label folder path
        all_files = os.listdir(args.label_folder_path)
        txt_files = list(filter(lambda x: x[-4:] == '.txt', all_files))

        #all_image_files = os.listdir(args.image_folder_path)
        #img_files = list(filter(lambda y: y[-4:] == '.jpg', all_files))
        
        #txt_files = glob.glob(args.label_folder_path + "*.txt")
        #print(len(txt_files))
        #print(len(img_files))

        for txt in txt_files:
            with open(args.label_folder_path + txt, 'rt') as current_file:
                img_name = txt[:-4] + '.jpg'
                img = PIL.Image.open(args.image_folder_path + img_name)

                width, height = img.size
                #print(str(width) + "x" + str(height))

                
                #print(img)
                #print(txt)
                
                data = current_file.read()
                lines = data.count('\n')
                #print(lines)
                data = data.split('\n')
                data = data[:-1]
                #print(data)
                
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
                # This is splitting detections to train, test, validate but there can be multiple detections on a single image, should I be separating them?

                for element in data:
                    detections = element.split()

                    detections[1] = round(float(detections[1]) / width, 4)
                    detections[2] = round(float(detections[2]) / height, 4)
                    detections[3] = round(float(detections[3]) / width, 4)
                    detections[4] = round(float(detections[4]) / height, 4)
                    
                    csv_writer.writerow([set_type,
                                         args.image_folder_path + img_name,
                                         detections[0], detections[1],
                                         detections[2], None, None,
                                         detections[3], detections[4],
                                         None, None])

                    #print(detections[0])

        # Open a .txt file, read its contents
        # Convert the annotations using the image size
        # Write the new annotations to a line in the csv
















    


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
