"""Converts OIDv4 annotations to match AutoML CSV format

 CSV format needed for the object_detector.Dataloader.from_csv()
 method as used in this article:
 https://www.tensorflow.org/lite/models/modify/model_maker/object_detection

 Run as:
     python OIDv4_to_csv.py /home/user/label_folder/ new_output.csv \
     /home/user/image_folder/
"""

import csv
import os
import PIL
import argparse
import random


def main():
    """Converts OIDv4 annotation data and writes to CSV

    Converts the OIDv4 denormalized xyxy bbox data to normalized
    xyxy format and writes the data to a CSV in the proper
    format [set, class, file, xmin, ymin, '', '', xmax, ymax, '', ''].
    """
    # pylint: disable=locally-disabled, too-many-locals
    # Get command line arguments when running program
    parser = argparse.ArgumentParser()
    parser.add_argument("label_folder_path", type=str,
                        help="path to folder containing txt input files")
    parser.add_argument("output_file", type=str,
                        help="filepath for the CSV output file")
    parser.add_argument("image_folder_path", type=str,
                        help="path to the image folder")
    args = parser.parse_args()

    with open(args.output_file, 'w', encoding='utf-8') as data_file:
        csv_writer = csv.writer(data_file)

        # Go to the label folder path and gather all text files
        all_files = os.listdir(args.label_folder_path)
        txt_files = list(filter(lambda x: x[-4:] == '.txt', all_files))

        # For each text file change the annotations and write to csv
        for txt in txt_files:
            with open(args.label_folder_path + txt, 'r') as current_file:
                img_name = txt[:-4] + '.jpg'
                img = PIL.Image.open(args.image_folder_path + img_name)

                width, height = img.size

                data = current_file.read()

                data = data.split('\n')
                data = data[:-1]

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

                # For each detection change to normalized xyxy
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


if __name__ == "__main__":
    main()
