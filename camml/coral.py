"""Tools for doing CV/ML on Google Coral devices for field cameras

"""
import time

import cv2
from PIL import Image

from pycoral.adapters import common, classify, detect
from pycoral.utils.edgetpu import make_interpreter


class ImageClassifierHandler():
    """Handles image classification on the coral

    """
    def __init__(self, model,
                 threshold=0.0,
                 top_k=5):
        self.interpreter = make_interpreter(model)
        self.interpreter.allocate_tensors()
        self.input_size = common.input_size(self.interpreter)

        self.threshold = threshold
        self.top_k = top_k

    def infer(self, frame):
        """Perform inference on image

        Parameters
        ----------
        frame : numpy.ndarray
            Image to perform inference on

        Returns
        -------
        results : list of pycoral.adapters.classify.Class
           classes identified in the image

        inference_time : float
           Time taken to perform inference in milliseconds
        """

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame).convert('RGB').resize(self.input_size,
                                                             Image.ANTIALIAS)
        common.set_input(self.interpreter, frame)

        start = time.perf_counter()
        self.interpreter.invoke()
        inference_time = (time.perf_counter() - start) * 1000

        classes = classify.get_classes(self.interpreter,
                                       self.top_k,
                                       self.threshold)

        return classes, inference_time


class ObjectDetectorHandler():
    """Handles object detection on the coral

    """

    def __init__(self,
                 model,
                 weights,  # not used.  here to match dnn libs' API
                 input_width,  # not used.  here to match dnn libs' API
                 input_height):  # not used.  here to match dnn libs' API

        self.interpreter = make_interpreter(model)
        self.interpreter.allocate_tensors()

    def infer(self, frame):
        """Perform object detection on image.

        Parameters
        ----------
        frame : numpy.ndarray
            Image on which to perform inference

        Returns
        -------
        results :

        inference_time : float
            Time taken to perform inference in milliseconds

        """

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        _, scale = common.set_resized_input(self.interpreter,
                                            frame.size,
                                            lambda size: frame.resize(size,
                                                                      Image.ANTIALIAS))

        start = time.perf_counter()
        self.interpreter.invoke()
        inf_time = (time.perf_counter() - start) * 1000
        objs = detect.get_objects(self.interpreter, 0.05, scale)

        return objs, inf_time

    def filter_boxes(self,
                     results,
                     frame,
                     confidence_threshold,
                     nms_threshold):
        """Apply filtering to boxes from inference

        Scan through all the bounding boxes from a network output from a
        network and keep only the ones with high confidence
        scores. Assign the box's class label as the class with the
        highest score.  Apply NMS on boxes.

        NOTE: Non-maximum suppression is not currently implemented

        Parameters
        ----------
        results :
            Objects from an inference

        frame : numpy.ndarray
            Frame the inference was performed on

        confidence_threshold : float
            Threshold for filtering boxes based on confidence score

        nms_threshold : float
            UNUSED!! Threshold for doing NMS filtering

        Returns
        -------
        list of dict:
            Each dict contains the class_id, confidence, and box
            dimensions for a given box.

        """

        class_ids = []
        confidences = []
        boxes = []

        for result in results:
            if result.score > confidence_threshold:
                class_ids.append(result.id)
                confidences.append(float(result.score))

                bbox = result.bbox
                boxes.append([bbox.xmin,
                              bbox.ymin,
                              bbox.xmax - bbox.xmin,
                              bbox.ymax - bbox.ymin])

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
    labels = []
    for row in open(label_file):
        labels.append(row.strip())

    return labels
    

# def read_classes_from_file(label_file):
#     """Read in class labels from a file.

#     Parameters
#     ----------
#     label_file : string
#         Filename of file that contains class ID with associated label
#         (one label per line). Class ID must match index numbers used
#         in neural net.

#     Returns
#     -------
#     labels : list of string
#         List of the class labels, indexable by class_id from network

#     """
#     ingested = []
#     highest_class_id = 0
#     for row in open(label_file):
#         class_id, label = row.strip().split(maxsplit=1)
#         class_id = int(class_id)
#         ingested.append([class_id, label.strip()])
#         if class_id > highest_class_id:
#             highest_class_id = class_id

#     labels = [None for i in range(highest_class_id + 1)]
#     for class_id, label in ingested:
#         labels[class_id] = label

#     return labels


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
