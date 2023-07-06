"""Converts megadetector output JSON file to match CSV format

 CSV format needed for the object_detector.Dataloader.from_csv()
 method as used in this article:
 https://www.tensorflow.org/lite/models/modify/model_maker/object_detection
"""

import json
import csv
import os


def main():
    """Converts JSON data and writes to CSV

    Converts the JSON xywh bbox data to xyxy format and writes the
    data to a CSV in the proper format [set, class, file, xmin, ymin,
    '', '', xmax, ymax, '', ''].
    """
    # Input JSON file
    input_file = 'test_output.json'

    # Output CSV file
    output_file = 'test_output.csv'

    # Opening JSON file and loading the data
    # into the variable data
    with open(input_file, encoding="utf-8") as json_file:
        data = json.load(json_file)

    image_data = data['images']

    # Open a file for writing
    with open(output_file, 'w', encoding="utf-8") as data_file:
        # Create the csv writer object
        csv_writer = csv.writer(data_file)

        for img in image_data:
            # Set the file path
            image_path = os.path.join('/home/user/test_images',
                                      img['file'])
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

                # Could randomly set 80% of images to train,
                # 10% to validation, and 10% to test. Currently all set
                # to UNASSIGNED then manually changed in csv
                if img['detections'][i]['category'] == '1':
                    csv_writer.writerow(['UNASSIGNED',
                                         image_path,
                                         'animal',
                                         img['detections'][i]['bbox'][0],
                                         img['detections'][i]['bbox'][1],
                                         None, None,
                                         img['detections'][i]['bbox'][2],
                                         img['detections'][i]['bbox'][3],
                                         None, None])
                elif img['detections'][i]['category'] == '2':
                    csv_writer.writerow(['UNASSIGNED',
                                         image_path,
                                         'person',
                                         img['detections'][i]['bbox'][0],
                                         img['detections'][i]['bbox'][1],
                                         None, None,
                                         img['detections'][i]['bbox'][2],
                                         img['detections'][i]['bbox'][3],
                                         None, None])
                elif img['detections'][i]['category'] == '3':
                    csv_writer.writerow(['UNASSIGNED',
                                         image_path,
                                         'vehicle',
                                         img['detections'][i]['bbox'][0],
                                         img['detections'][i]['bbox'][1],
                                         None, None,
                                         img['detections'][i]['bbox'][2],
                                         img['detections'][i]['bbox'][3],
                                         None, None])
                else:
                    csv_writer.writerow(['UNASSIGNED',
                                         image_path,
                                         '', '', '', '', '',
                                         '', '', '', ''])

        # To make images with no detections appear in the csv file
        # uncomment if block
        # if (len(img['detections']) == 0):
        #    csv_writer.writerow(['UNASSIGNED',
        #                         image_path,
        #                         None, None, None, None,
        #                         None, None, None, None, None])


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
