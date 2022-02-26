import torch
import torchvision
from models.evaluators.nms import non_max_suppression


def coco_post(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    """
    The prediction is [center_x, center_y, h, w]
    Turn it to [x1, y1, x2, y2] because the NMS function need this.
    :param prediction: [batch, num_prediction, prediction]
        predicted box format: (center x, center y, w, h)
    :param num_classes:
    :param conf_thre:
    :param nms_thre:
    :param class_agnostic:
    :return: (x1, y1, x2, y2, obj_conf, class_conf, class)
    """
    if isinstance(prediction, list):
        prediction = torch.cat(prediction, 1)

    # from (cx,cy,w,h) to (x1,y1,x2,y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thre).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with the highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)

        # NMS processed with class
        unique_labels = detections[:, -1].unique()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get lightning with the highest confidence and save as max lightning
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last lightning
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thre]

            max_detections = torch.cat(max_detections)
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None \
                else torch.cat((output[image_i], max_detections))

        # if not detections.size(0):
        #     continue
        #
        # # NMS
        # if class_agnostic:
        #     nms_out_index = torchvision.ops.nms(
        #         detections[:, :4],
        #         detections[:, 4] * detections[:, 5],
        #         nms_thre,
        #     )
        # else:
        #     nms_out_index = torchvision.ops.batched_nms(
        #         detections[:, :4],
        #         detections[:, 4] * detections[:, 5],
        #         detections[:, 6],
        #         nms_thre,
        #     )
        # # detections = non_max_suppression
        # detections = detections[nms_out_index]
        # output[i] = detections
    return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
