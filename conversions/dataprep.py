"""Data prep tools

Currently, is simply functions required by various of the conversion
scripts in this folder.

"""


def write_detection_to_csv(csv_writer, detection, set_type,
                           image_path, class_label,
                           confidence_threshold=.15):
    # pylint: disable=too-many-arguments

    """Write positive detection into CSV file

    Megadetector uses 3 categories 1-animal, 2-person, 3-vehicle.
    This camml training workflow assumes that we only want the animal
    detections in the training set so this function filters out the
    other two categories. The 'confidence_threshold' argument further
    filters the detections so only >= conf detections appear.  One
    might try to tune this threshold as a hyperparameter of training,
    the goal being to get as much "good" data in as possible while not
    using "bad" MegaDetector detections in training.

    """
    if (detection['category'] == '1'
       and detection['conf'] >= confidence_threshold):
        csv_writer.writerow([set_type,
                             image_path,
                             class_label,
                             detection['bbox'][0],
                             detection['bbox'][1],
                             None, None,
                             detection['bbox'][2],
                             detection['bbox'][3],
                             None, None])


def bbox_to_pascal(bbox):
    """Convert xywh bbox to xmin, ymin, xmax, ymax bbox

    """
    x_min, y_min, x_max, y_max = coco_to_pascal_voc(bbox[0],
                                                    bbox[1],
                                                    bbox[2],
                                                    bbox[3])

    # Round to 4 digits to match CSV format
    x_min, y_min = round(x_min, 4), round(y_min, 4)
    x_max, y_max = round(x_max, 4), round(y_max, 4)

    # Set bbox coordinates to new values
    bbox[0] = x_min
    bbox[1] = y_min
    bbox[2] = x_max
    bbox[3] = y_max


def coco_to_pascal_voc(x_tl, y_tl, width, height):
    """Convert Coco bounding box to Pascal VOC bounding box

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
