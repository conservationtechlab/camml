"""Tools for doing CV/ML on Google Coral devices for field cameras

"""
import time

import cv2
import imutils

from PIL import Image
from edgetpu.detection.engine import DetectionEngine
from edgetpu.classification.engine import ClassificationEngine
# from edgetpu.utils import dataset_utils


class ImageClassifierHandler():

    def __init__(self, model):
        self.engine = ClassificationEngine(model)

    def infer(self, frame):
        """Infer class of image.

        Parameters
        ----------
        frame : numpy.ndarray
            Image to perform inference on

        Returns
        -------
        results :
           top_1 classification

        inference_time : float
           Time taken to perform inference in milliseconds
        """

        frame = imutils.resize(frame, width=500)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        start = time.time()
        results = self.engine.classify_with_image(frame, top_k=1)
        end = time.time()

        inference_time = (end-start) * 1000.0

        return results, inference_time


class ObjectDetectorHandler():

    def __init__(self,
                 model,
                 weights,  # not used.  here to match dnn libs' API
                 input_width,  # not used.  here to match dnn libs' API
                 input_height):  # not used.  here to match dnn libs' API

        self.DETECTOR_FRAME_WIDTH = 500
        self.engine = DetectionEngine(model)
        # self.labels = dataset_utils.read_label_file(labels)

    def infer(self, frame):
        """Perform object detection on image.

        Parameters
        ----------
        frame : numpy.ndarray
            Image on which to perform inference

        Returns
        -------
        results

        inference_time

        """
        frame = imutils.resize(frame, width=self.DETECTOR_FRAME_WIDTH)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        start = time.time()
        results = self.engine.detect_with_image(frame,
                                                threshold=0.05,
                                                # keep_aspect_ratio=args.keep_aspect_ratio,
                                                keep_aspect_ratio=True,
                                                relative_coord=False,
                                                top_k=10)
        end = time.time()
        inference_time = (end-start) * 1000.0
        return results, inference_time

    def filter_boxes(self,
                     results,
                     frame,
                     confidence_threshold,
                     nms_threshold):
        """Scan through all the bounding boxes that are output from a network
        and keep only the ones with high confidence scores. Assign the
        box's class label as the class with the highest score.  Apply NMS
        on boxes.

        Return a list of dicts, each dict containing the class_id,
        confidence, and box dimensions for a given box.

        """

        class_ids = []
        confidences = []
        boxes = []

        ratio = frame.shape[1]/self.DETECTOR_FRAME_WIDTH

        for result in results:
            if result.score > confidence_threshold:
                class_ids.append(result.label_id)
                confidences.append(float(result.score))

                box = result.bounding_box.flatten().astype('int')
                (startX, startY, endX, endY) = box
                boxes.append([int(startX * ratio),
                              int(startY * ratio),
                              int((endX - startX) * ratio),
                              int((endY - startY) * ratio)])

        # TODO: Do non-maximum suppression on the boxes

        lboxes = []

        # hack for while NMS thresholding isn't set up
        indices = range(len(boxes))

        for i in indices:
            lbox = {'class_id': class_ids[i],
                    'confidence': float(confidences[i]),
                    'box': boxes[i]}
            lboxes.append(lbox)

        return lboxes


def read_classes_from_file(label_file):
    """Read in class labels from a file.

    Parameters
    ----------
    label_file : string
        Filename of file that contains class ID with associated label
        (one label per line). Class ID must match index numbers used
        in neural net.

    Returns
    -------
    labels : list of string
        List of the class labels, indexable by class_id from network

    """
    ingested = []
    highest_class_id = 0
    for row in open(label_file):
        class_id, label = row.strip().split(maxsplit=1)
        class_id = int(class_id)
        ingested.append([class_id, label.strip()])
        if class_id > highest_class_id:
            highest_class_id = class_id

    labels = [None for i in range(highest_class_id + 1)]
    for class_id, label in ingested:
        labels[class_id] = label

    return labels


class TargetDetector(ObjectDetectorHandler):
    def __init__(self,
                 model_config,
                 model_weights,
                 input_width,
                 input_height,
                 conf_threshold,
                 nms_threshold,
                 class_names,
                 target_class):

        super().__init__(model_config,
                         model_weights,
                         input_width,
                         input_height)

        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.target_class = target_class
        self.class_names = class_names

    def detect(self, frame):
        results, inference_time = self.infer(frame)
        msg = ("[INFO] Inference time: "
               + "{:.1f} milliseconds".format(inference_time))
        print(msg)
        lboxes = self.filter_boxes(results,
                                   frame,
                                   self.conf_threshold,
                                   self.nms_threshold)

        # extract the lbox with the highest confidence (that is a target type)
        highest_confidence_target_class = 0
        target_lbox = None
        for lbox in lboxes:
            if self.class_names[lbox['class_id']] in self.target_class:
                if lbox['confidence'] > highest_confidence_target_class:
                    highest_confidence_target_class = lbox['confidence']
                    target_lbox = lbox

        return target_lbox
