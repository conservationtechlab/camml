"""Tools for drawing graphics related to camml

Classes and functions related to drawing the graphics that help
visualize the outputs and actions of elements of the camml package.

"""
import logging

import cv2


def labeled_box(frame,
                classes,
                lbox,
                thickness=1,
                color=(187, 209, 136),
                show_score=True,
                font_size=1.5,
                font_thickness=4,
                draw_label=True):
    """Draw a bounding box with info label

    Bounding box has label attached that displays the class ID and
    confidence

    """
    
    box = lbox['box']

    left, top, right, bottom = box_to_coords(box, return_kind='corners')

    logging.debug('Drawing box')
    cv2.rectangle(frame,
                  (left, top),
                  (right, bottom),
                  color,
                  thickness=thickness)

    if draw_label:
        label_props = {'font_size': font_size,
                       'font_thickness': font_thickness,
                       'show_score': show_score,
                       'top': bottom,
                       'left': left}

        _draw_label(frame,
                    classes,
                    lbox,
                    label_props)


def box_to_coords(box,
                  return_kind='corner_with_dims'):
    """Return coordinates of corner points of box.

    Parameters
    ----------

    box : list of int 
        Contains the 4 numbers that describe the box in pixel-space in
        the order of x coordinate of upper left corner, y coordinate
        of upper left corner, width of box, height of box

    return_kind : string
        "corner_with_dims": return left, top, width, height
        "corners": return left, top, right, and bottom
        "center": return x_center, y_center

    Returns
    -------
    
    Box coordinated re-computed based on return_kind

    """

    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]

    if return_kind == 'corners':
        right = left + width
        bottom = top + height
        return left, top, right, bottom
    elif return_kind == 'center':
        xc = left + width/2
        yc = top + height/2
        return xc, yc
    else:
        return left, top, width, height


def _draw_label(frame,
                classes,
                lbox,
                props):

    confidence = lbox['confidence']
    
    if props['show_score']:
        label = ':{:.2f}'.format(confidence)
    else:
        label = ''

    # Get the label for the class name and its confidence
    if classes:
        class_id = lbox['class_id']
        assert(class_id < len(classes))
        label = '{}{}'.format(classes[class_id], label)
    elif 'class_name' in lbox:
        label = '{}{}'.format(lbox['class_name'], label)

    label_size, base_line = cv2.getTextSize(label,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            props['font_size'],
                                            props['font_thickness'])

    cv2.rectangle(frame,
                  (props['left'], props['top']),
                  (props['left'] + round(1.05*label_size[0]),
                   props['top'] + round(1.5*label_size[1])),
                  (255, 255, 255),
                  cv2.FILLED)

    cv2.putText(frame,
                label,
                ((props['left'] + 10),
                 props['top'] + round(1.2*label_size[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                props['font_size'],
                (0, 0, 0),
                props['font_thickness'])
