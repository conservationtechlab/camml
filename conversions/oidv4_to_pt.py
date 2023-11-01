"""Converts OIDv4 annotations to match Yolov8 pytorch format

A .yaml file with image directory and class information is needed to
train a Yolov8 model. The yaml file will need a format similar to
the one described in the "EXAMPLE" section here:
https://roboflow.com/formats/yolov8-pytorch-txt

  Run as:
      python oidv4_to_pt.py /home/user/oidv4_labels/ \
      /home/user/output_labels/
"""

import argparse
import os

from PIL import Image


def main():
    """Converts OIDv4 annotation data and writes to txt files

    Converts the OIDv4 denormalized xyxy bbox data to centerx, centery,
    width, height and writes the data to txt files in the proper format.
    """
    # pylint: disable=locally-disabled, too-many-locals
    # Get command line arguments when running program
    parser = argparse.ArgumentParser()
    parser.add_argument("label_folder_path", type=str,
                        help="path to folder containing txt input files")
    parser.add_argument("output_folder_path", type=str,
                        help="path to folder to contain output files")
    parser.add_argument("image_folder_path", type=str,
                        help="path to the image folder")
    args = parser.parse_args()

    # Go to the label folder path and gather all text files
    all_files = os.listdir(args.label_folder_path)
    txt_files = list(filter(lambda x: x[-4:] == '.txt', all_files))

    # For each text file convert annotations and write to a new text file
    for txt in txt_files:
        with open(args.label_folder_path + txt, 'r',
                  encoding='utf-8') as current_file:
            img_name = txt[:-4] + '.jpg'
            img = Image.open(args.image_folder_path + img_name)

            img_width, img_height = img.size

            data = current_file.read()

            data = data.split('\n')
            data = data[:-1]

        with open(args.output_folder_path + txt, 'w+',
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

                # Separate detections onto different lines
                text_file.seek(0)
                first_char = text_file.read(1)
                if not first_char:
                    line = '0 ' + str(center_x) + ' ' \
                           + str(center_y) + ' ' + str(bbox_width) \
                           + ' ' + str(bbox_height)
                    text_file.write(line)
                else:
                    line = '\n0 ' + str(center_x) + ' ' \
                           + str(center_y) + ' ' + str(bbox_width) \
                           + ' ' + str(bbox_height)
                    text_file.write(line)


if __name__ == "__main__":
    main()
