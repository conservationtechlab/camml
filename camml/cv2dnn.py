"""Tools for CV/ML for camera systems using OpenCV dnn module

"""

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class ObjectDetectorHandler():
    """Handler for cv2.dnn object detection neural network, using
    specifically YOLO (may not generalize to other object detectors)

    Attributes
    ----------
    network : cv2.dnn_Net
       Darknet Yolo-style neural network

    input_width : int
        Width of images for inference

    input_height : int
        Height of images for inference

    """

    def __init__(self,
                 model_config,
                 model_weights,
                 input_width,
                 input_height):
        """
        Parameters
        ----------
        model_config : string
            The path of model configuration file

        model_weights : string
            The path of file that has model weights

        input_width : int
            Width of images for inference

        input_height : int
            Height of images for inference

        """

        self.model = cv2.dnn.readNetFromDarknet(model_config,
                                                model_weights)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.input_width = input_width
        self.input_height = input_height

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
        blob = cv2.dnn.blobFromImage(frame,
                                     1/255,
                                     (self.input_width,
                                      self.input_height),
                                     [0, 0, 0],
                                     1,
                                     crop=False)

        self.model.setInput(blob)
        results = self.model.forward(get_outputs_names(self.model))

        overall_time, layers_times = self.model.getPerfProfile()
        inference_time = overall_time * 1000.0 / cv2.getTickFrequency()

        return results, inference_time

    def filter_boxes(self,
                     results,
                     frame,
                     confidence_threshold,
                     nms_threshold):
        """Threshold boxes from inference.

        Scan through all the bounding boxes that are output from a
        network and keep only the ones with high confidence
        scores. Assign the box's class label as the class with the
        highest score.  Apply NMS on boxes.

        Return a list of dicts, each dict containing the class_id,
        confidence, and box dimensions for a given box.

        """

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        class_ids = []
        confidences = []
        boxes = []

        for result in results:
            for detection in result:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Do non-maximum suppression on the boxes
        filtered_box_indices = cv2.dnn.NMSBoxes(boxes,
                                                confidences,
                                                confidence_threshold,
                                                nms_threshold)
        lboxes = []

        for i in filtered_box_indices:
            lbox = {'class_id': class_ids[i],
                    'confidence': float(confidences[i]),
                    'box': boxes[i]}
            lboxes.append(lbox)

        return lboxes


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
        msg = ("Inference time: "
               + "{:.1f} milliseconds".format(inference_time))
        log.debug(msg)
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


def read_classes_from_file(classes_file):
    """Read in class names from a file.

    Parameters
    ----------
    classes_file : string
        Filename of file that contains classes (one per line).  Order
        must match index numbers used in neural net.

    Returns
    -------
    classes : list of string
        List of the class names, indexable by class_id from network

    """
    with open(classes_file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes


def get_outputs_names(network):
    """Get the names of the output layers of the neural net

    Parameters
    ----------

    network :
       The neural net whose output names you want

    Returns
    -------

    """

    # Get the names of all the layers in the network
    layer_names = network.getLayerNames()
    # Get the names of the output layers, i.e. the layers with
    # unconnected outputs
    return [layer_names[i - 1] for i in network.getUnconnectedOutLayers()]
