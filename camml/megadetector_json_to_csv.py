'''
 Convert megadetector output JSON file to match CSV format
 for training object detection models.

 CSV format needed for the object_detector.Dataloader.from_csv()
 method as used in this article:
 https://www.tensorflow.org/lite/models/modify/model_maker/object_detection
'''

import json
import csv


def main():
    # Opening JSON file and loading the data
    # into the variable data
    with open('test_output.json') as json_file:
        data = json.load(json_file)

    box_data = data['images']
    image_data = data['images']

    for box in box_data:
        for i in range(0, len(box['detections'])):
            # Set xywh from bbox data
            x = box['detections'][i]['bbox'][0]
            y = box['detections'][i]['bbox'][1]
            w = box['detections'][i]['bbox'][2]
            h = box['detections'][i]['bbox'][3]

            # Convert xywh bbox to xmin, ymin, xmax, ymax bbox
            x_min, y_min, x_max, y_max = coco_to_pascal_voc(x, y, w, h)

            # Round to 4 digits to match CSV format
            x_minr, y_minr = round(x_min, 4), round(y_min, 4)
            x_maxr, y_maxr = round(x_max, 4), round(y_max, 4)

            # Set bbox coordinates to new values
            box['detections'][i]['bbox'][0] = x_minr
            box['detections'][i]['bbox'][1] = y_minr
            box['detections'][i]['bbox'][2] = x_maxr
            box['detections'][i]['bbox'][3] = y_maxr

    # Open a file for writing
    data_file = open('test_output.csv', 'w')

    # Create the csv writer object
    csv_writer = csv.writer(data_file)

    for img in image_data:
        # Writing data of CSV file
        for n in range(0, len(img['detections'])):
            # Could randomly set 80% of images to train, 10% to validation,
            # and 10% to test
            # Currently all set to UNASSIGNED then manually changed in csv
            if (img['detections'][n]['category'] == '1'):
                csv_writer.writerow(['UNASSIGNED',
                                     '/home/user/test_images/' + img['file'],
                                     'animal',
                                     img['detections'][n]['bbox'][0],
                                     img['detections'][n]['bbox'][1],
                                     None, None,
                                     img['detections'][n]['bbox'][2],
                                     img['detections'][n]['bbox'][3],
                                     None, None])
            elif (img['detections'][n]['category'] == '2'):
                csv_writer.writerow(['UNASSIGNED',
                                     '/home/user/test_images/' + img['file'],
                                     'person',
                                     img['detections'][n]['bbox'][0],
                                     img['detections'][n]['bbox'][1],
                                     None, None,
                                     img['detections'][n]['bbox'][2],
                                     img['detections'][n]['bbox'][3],
                                     None, None])
            elif (img['detections'][n]['category'] == '3'):
                csv_writer.writerow(['UNASSIGNED',
                                     '/home/user/test_images/' + img['file'],
                                     'vehicle',
                                     img['detections'][n]['bbox'][0],
                                     img['detections'][n]['bbox'][1],
                                     None, None,
                                     img['detections'][n]['bbox'][2],
                                     img['detections'][n]['bbox'][3],
                                     None, None])
            else:
                csv_writer.writerow(['UNASSIGNED',
                                     '/home/user/test_images/' + img['file'],
                                     '', '', '', '', '', '', '', '', ''])

    # To make images with no detections appear in the csv file
    # uncomment if block
    # if (len(img['detections']) == 0):
    #    csv_writer.writerow(['UNASSIGNED',
    #                         '/home/user/test_images/' + img['file'],
    #                         None, None, None, None,
    #                         None, None, None, None, None])


# Convert Coco bb to Pascal_Voc bb
def coco_to_pascal_voc(x1, y1, w, h):
    return [x1, y1, x1 + w, y1 + h]


if __name__ == "__main__":
    main()
