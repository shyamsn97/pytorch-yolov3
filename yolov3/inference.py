from collections import deque
import colorsys
import threading
import time

import cv2
import numpy as np
import torch
import os

CLASS_NAMES = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush',
]

def unique_colors(num_colors):
    """
    Yield `num_colors` unique BGR colors. Uses HSV space as intermediate.

    Args:
        num_colors (int): Number of colors to yield.

    Yields:
        3-tuple of 8-bit BGR values.
    """
    for H in np.linspace(0, 1, num_colors, endpoint=False):
        rgb = colorsys.hsv_to_rgb(H, 1.0, 1.0)
        bgr = (int(255 * rgb[2]), int(255 * rgb[1]), int(255 * rgb[0]))
        yield bgr


def draw_boxes(
    img, bbox_tlbr, class_prob=None, class_idx=None, class_names=None
):
    """
    Draw bboxes (and class names or indices for each bbox) on an image.
    Bboxes are drawn in-place on the original image.

    If `class_prob` is provided, the prediction probability for each bbox
    will be displayed along with the bbox. If `class_idx` is provided, the
    class index of each bbox will be displayed along with the bbox. If both
    `class_idx` and `class_names` are provided, `class_idx` will be used to
    determine the class name for each bbox and the class name of each bbox
    will be displayed along with the bbox.

    If `class_names` is provided, a unique color is used for each class.

    Args:
        img (np.ndarray): Image on which to draw bboxes.
        bbox_tlbr (np.ndarray): Mx4 array of M detections.
        class_prob (np.ndarray): Array of M elements corresponding to predicted
            class probabilities for each bbox.
        class_idx (np.ndarray): Array of M elements corresponding to the
            class index with the greatest probability for each bbox.
        class_names (list): List of all class names in order.
    """
    colors = None
    if class_names is None:
        class_names = CLASS_NAMES
        
    if class_names is not None:
        colors = dict()
        num_colors = len(CLASS_NAMES)
        colors = list(unique_colors(num_colors))
    
    for i, (tl_x, tl_y, br_x, br_y) in enumerate(bbox_tlbr):
        if CLASS_NAMES[class_idx[i]] in class_names:
            bbox_text = []
            
            if colors is not None:
                color = colors[class_idx[i]]
            else:
                color = (0, 255, 0)

            bbox_text.append(CLASS_NAMES[class_idx[i]])
            if class_idx is not None:
                bbox_text.append(str(class_idx[i]))

            if class_prob is not None:
                bbox_text.append("({:.2f})".format(class_prob[i]))

            bbox_text = " ".join(bbox_text)
            color = tuple(map(int, color))
            cv2.rectangle(
                img, (tl_x, tl_y), (br_x, br_y), (30,15,130), 2
            )

            if bbox_text:
                cv2.rectangle(
                    img, (tl_x + 1, tl_y + 1),
                    (tl_x + int(8 * len(bbox_text)), tl_y + 18),
                    (20, 20, 20), cv2.FILLED
                )
                cv2.putText(
                    img, bbox_text, (tl_x + 1, tl_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
                )


def _non_max_suppression(bbox_tlbr, prob, iou_thresh=0.3):
    """
    Perform non-maximum suppression on an array of bboxes and return the
    indices of detections to retain.

    Derived from:
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    Args:
        bbox_tlbr (np.ndarray): An Mx4 array of bboxes (consisting of M
            detections/bboxes), where bbox_tlbr[:, :4] represent the four
            bbox coordinates.
        prob (np.ndarray): An array of M elements corresponding to the
            max class probability of each detection/bbox.

    Returns:
        List of bbox indices to keep (ie, discard everything except
        `bbox_tlbr[idxs_to_keep]`).
    """

    # Compute area of each bbox.
    area = (
        ((bbox_tlbr[:, 2] - bbox_tlbr[:, 0]) + 1)
        * ((bbox_tlbr[:, 3] - bbox_tlbr[:, 1]) + 1)
    )

    # Sort detections by probability (largest to smallest).
    idxs = deque(np.argsort(prob)[::-1])
    idxs_to_keep = list()

    while idxs:
        # Grab current index (index corresponding to the detection with the
        # greatest probability currently in the list of indices).
        curr_idx = idxs.popleft()
        idxs_to_keep.append(curr_idx)

        # Find the coordinates of the regions of overlap between the current
        # detection and all other detections.
        overlaps_tl_x = np.maximum(bbox_tlbr[curr_idx, 0], bbox_tlbr[idxs, 0])
        overlaps_tl_y = np.maximum(bbox_tlbr[curr_idx, 1], bbox_tlbr[idxs, 1])
        overlaps_br_x = np.minimum(bbox_tlbr[curr_idx, 2], bbox_tlbr[idxs, 2])
        overlaps_br_y = np.minimum(bbox_tlbr[curr_idx, 3], bbox_tlbr[idxs, 3])

        # Compute width and height of overlapping regions.
        overlap_w = np.maximum(0, (overlaps_br_x - overlaps_tl_x) + 1)
        overlap_h = np.maximum(0, (overlaps_br_y - overlaps_tl_y) + 1)

        # Compute amount of overlap (intersection).
        inter = overlap_w * overlap_h
        union = area[curr_idx] + area[idxs] - inter
        iou = inter / union

        idxs_to_remove = [idxs[i] for i in np.where(iou > iou_thresh)[0]]
        for idx in idxs_to_remove:
            idxs.remove(idx)

    return idxs_to_keep


def non_max_suppression(bbox_tlbr, class_prob, class_idx=None, iou_thresh=0.3):
    """
    Perform non-maximum suppression (NMS) of bounding boxes. If `class_idx` is
    provided, per-class NMS is performed by performing NMS on each class and
    combining the results. Else, bboxes are suppressed without regard for
    class.

    Args:
        bbox_tlbr (np.ndarray): Mx4 array of M bounding boxes, where dim 1
            indices are: top left x, top left y, bottom right x, bottom
            right y.
        class_prob (np.ndarray): Array of M elements corresponding to predicted
            class probabilities for each bbox.
        class_idx (np.ndarray): Array of M elements corresponding to the
            class index with the greatest probability for each bbox. If
            provided, per-class NMS is performed; else, all bboxes are
            treated as a single class.
        iou_thresh (float): Intersection over union (IOU) threshold for
            bbox to be considered a duplicate. 0 <= `iou_thresh` < 1.

    Returns:
        List of bbox indices to keep (ie, discard everything except
        `bbox_tlbr[idxs_to_keep]`).
    """

    if class_idx is not None:
        # Perform per-class non-maximum suppression.
        idxs_to_keep = []

        # Set of unique class indices.
        unique_class_idxs = set(class_idx)

        for class_ in unique_class_idxs:
            # Bboxes corresponding to the current class index.
            curr_class_mask = np.where(class_idx == class_)[0]
            curr_class_bbox = bbox_tlbr[curr_class_mask]
            curr_class_prob = class_prob[curr_class_mask]

            curr_class_idxs_to_keep = _non_max_suppression(
                curr_class_bbox, curr_class_prob, iou_thresh
            )
            idxs_to_keep.extend(
                curr_class_mask[curr_class_idxs_to_keep].tolist()
            )
    else:
        idxs_to_keep = _non_max_suppression(bbox_tlbr, class_prob, iou_thresh)
    return idxs_to_keep


def cxywh_to_tlbr(bbox_xywh):
    """
    Args:
        bbox_xywh (np.array): An MxN array of detections where bbox_xywh[:, :4]
            correspond to coordinates (center x, center y, width, height).

    Returns:
        An MxN array of detections where bbox_tlbr[:, :4] correspond to
        coordinates (top left x, top left y, bottom right x, bottom right y).
    """

    bbox_tlbr = np.copy(bbox_xywh)
    bbox_tlbr[:, :2] = bbox_xywh[:, :2] - (bbox_xywh[:, 2:4] // 2)
    bbox_tlbr[:, 2:4] = bbox_xywh[:, :2] + (bbox_xywh[:, 2:4] // 2)
    return bbox_tlbr


def inference(
    net, images, device="cuda", prob_thresh=0.05, nms_iou_thresh=0.3,
    resize=True
):
    """
    Run inference on image(s) and return the corresponding bbox coordinates,
    bbox class probabilities, and bbox class indices.

    Args:
        net (torch.nn.Module): Instance of network class.
        images (List[np.ndarray]): List (batch) of images to process
            simultaneously.
        device (str): Device for inference (eg, "cpu", "cuda").
        prob_thresh (float): Probability threshold for detections to keep.
            0 <= prob_thresh < 1.
        nms_iou_thresh (float): Intersection over union (IOU) threshold for
            non-maximum suppression (NMS). Per-class NMS is performed.
        resize (bool): If True, resize image(s) to dimensions given by the
            `net_info` attribute/block of `net` (from the Darknet .cfg file)
            before pushing through network.

    Returns:
        List of lists (one for each image in the batch) of:
            bbox_tlbr (np.ndarray): Mx4 array of bbox top left/bottom right
                coords.
            class_prob (np.ndarray): Array of M predicted class probabilities.
            class_idx (np.ndarray): Array of M predicted class indices.
    """
    if not isinstance(images, list):
        images = [images]

    orig_image_shapes = [image.shape for image in images]

    # Resize input images to match shape of images on which net was trained.
    if resize:
        net_image_shape = (net.net_info["height"], net.net_info["width"])
        images = [
            cv2.resize(image, net_image_shape)
            if image.shape[:2] != net_image_shape
            else image for image in images
        ]

    # Stack images along new batch axis, flip channel axis so channels are RGB
    # instead of BGR, transpose so channel axis comes before row/column axes,
    # and convert pixel values to FP32. Do this in one step to ensure array
    # is contiguous before passing to torch tensor constructor.
    inp = np.transpose(np.flip(np.stack(images), 3), (0, 3, 1, 2)).astype(
        np.float32) / 255.0

    inp = torch.tensor(inp, device=device)
    out = net.forward(inp)

    bbox_xywh = out["bbox_xywh"].detach().cpu().numpy()
    class_prob = out["class_prob"].cpu().numpy()
    class_idx = out["class_idx"].cpu().numpy()

    thresh_mask = class_prob >= prob_thresh

    # Perform post-processing on each image in the batch and return results.
    results = []
    for i in range(bbox_xywh.shape[0]):
        image_bbox_xywh = bbox_xywh[i, thresh_mask[i, :], :]
        image_class_prob = class_prob[i, thresh_mask[i, :]]
        image_class_idx = class_idx[i, thresh_mask[i, :]]

        image_bbox_xywh[:, [0, 2]] *= orig_image_shapes[i][1]
        image_bbox_xywh[:, [1, 3]] *= orig_image_shapes[i][0]
        image_bbox_tlbr = cxywh_to_tlbr(image_bbox_xywh.astype(np.int))

        idxs_to_keep = non_max_suppression(
            image_bbox_tlbr, image_class_prob, class_idx=image_class_idx,
            iou_thresh=nms_iou_thresh
        )

        results.append(
            [
                image_bbox_tlbr[idxs_to_keep, :],
                image_class_prob[idxs_to_keep],
                image_class_idx[idxs_to_keep]
            ]
        )

    return results


def to_coco(image_filenames, inference_output, class_names):
    """
    Convert output from `inference()` to a COCO dataset.

    Args:
        image_filenames (List[str]): list of image filenames corresponding to
            each element of `inference_output`.
        inference_output (list): List of (bbox_xywh, class_prob, class_idx)
            tuples for each image processed by `inference()`.
        class_names (List[str]): List of class names.

    Returns:
        Dict representing a COCO object detection dataset.
    """
    categories = []
    for i, class_name in enumerate(class_names):
        categories.append(
            {
                "id": i,
                "name": class_name
            }
        )

    dataset = {
        "info": [],
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    num_annotations = 0
    for i, image_output in enumerate(inference_output):
        bbox_tlbr, class_prob, class_idx = image_output

        # Assign an arbitrary id to the image.
        image = {
            "file_name": image_filenames[i],
            "id": i,
        }
        dataset["images"].append(image)

        for j, (tl_x, tl_y, br_x, br_y) in enumerate(bbox_tlbr):
            tl_x = int(tl_x)
            tl_y = int(tl_y)
            br_x = int(br_x)
            br_y = int(br_y)
            w = br_x - tl_x
            h = br_y - tl_y

            ann = {
                "image_id": i,
                "bbox": [tl_x, tl_y, w, h],
                "category_id": class_idx[j],
                "id": num_annotations,
                "score": float(class_prob[j]),
                "area": w * h,
            }
            dataset["annotations"].append(ann)
            num_annotations += 1

    return dataset


def detect_in_cam(
    net, cam_id=0, device="cuda", prob_thresh=0.05, nms_iou_thresh=0.3,
    class_names=None, show_fps=False, frames=None
):
    """
    Run and display real-time inference on a webcam stream.

    Performs inference on a webcam stream, draw bounding boxes on the frame,
    and display the resulting video in real time.

    Args:
        net (torch.nn.Module): Instance of network class.
        cam_id (int): Camera device id.
        device (str): Device for inference (eg, "cpu", "cuda").
        prob_thresh (float): Detection probability threshold.
        nms_iou_thresh (float): NMS IOU threshold.
        class_names (list): List of all model class names in order.
        show_fps (bool): Display current frames processed per second.
        frames (list): Optional list to populate with frames being displayed;
            can be used to write or further process frames after this function
            completes. Because mutables (like lists) are passed by reference
            and are modified in-place, this function has no return value.
    """

    # Number of frames to average for computing FPS.
    num_fps_frames = 30
    previous_fps = deque(maxlen=num_fps_frames)
    cam = cv2.VideoCapture(0)
    print("CLASS NAMES: {}".format(class_names))
    while True:
        loop_start_time = time.time()
        ret_val, frame = cam.read()

        bbox_tlbr, probs, class_idx = inference(
            net, frame, device=device, prob_thresh=prob_thresh,
            nms_iou_thresh=nms_iou_thresh
        )[0]
        draw_boxes(
            frame, bbox_tlbr, class_idx=class_idx, class_names=class_names, class_prob=probs
        )

        if show_fps:
            cv2.putText(
                frame,  f"{int(sum(previous_fps) / num_fps_frames)} fps",
                (2, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9,
                (255, 255, 255)
            )

        # video_shower.frame = frame
        if frames is not None:
            frames.append(frame)

        previous_fps.append(int(1 / (time.time() - loop_start_time)))
        cv2.imshow("my_webcam", frame)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def detect_in_video(
    net, filepath, device="cuda", prob_thresh=0.05, nms_iou_thresh=0.3,
    class_names=None, frames=None, show_video=True
):
    """
    Run and optionally display inference on a video file.

    Performs inference on a video, draw bounding boxes on the frame,
    and optionally display the resulting video.

    Args:
        net (torch.nn.Module): Instance of network class.
        filepath (str): Path to video file.
        device (str): Device for inference (eg, "cpu", "cuda").
        prob_thresh (float): Detection probability threshold.
        nms_iou_thresh (float): NMS IOU threshold.
        cam_id (int): Camera device id.
        class_names (list): List of all model class names in order.
        frames (list): Optional list to populate with frames being displayed;
            can be used to write or further process frames after this function
            completes. Because mutables (like lists) are passed by reference
            and are modified in-place, this function has no return value.
        show_video (bool): Whether to display output while processing.
    """
    cap = cv2.VideoCapture(filepath)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        bbox_tlbr, _, class_idx = inference(
            net, frame, device=device, prob_thresh=prob_thresh,
            nms_iou_thresh=nms_iou_thresh
        )[0]
        draw_boxes(
            frame, bbox_tlbr, class_idx=class_idx, class_names=class_names
        )

        if frames is not None:
            frames.append(frame)

        if show_video:
            cv2.imshow("YOLOv3", frame)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
