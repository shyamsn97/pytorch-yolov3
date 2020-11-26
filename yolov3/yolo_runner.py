from yolov3.darknet import Darknet
from yolov3.inference import CLASS_NAMES, non_max_suppression, cxywh_to_tlbr, draw_boxes
import time
import warnings
import torch
import os
import cv2
from collections import deque
import numpy as np
import typing

class YoloRunner:

    def __init__(self, 
                weight_path: str, # Path to Darknet model weights file
                model_config: str, #  Path to Darknet model config file
                class_names:typing.List[str] = CLASS_NAMES, # classes to classify
                image_path: str = None, # path to images
                video_path: str = None, # path to video
                device: str = "cpu",
                prob_threshold: float = 0.05,
                nms_iou_thresh: float = 0.3,
                show_fps: bool = True,
                verbose: bool = True,
                output_path: str = None,
                ):
        self.device = device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                "CUDA not available; falling back to CPU. Pass `-d cpu` or ensure "
                "compatible versions of CUDA and pytorch are installed.",
                RuntimeWarning, stacklevel=2
            )
            self.device = "cpu"
        # Expand pathlib Paths and convert to string.
        path_args = [
            model_config, weight_path, image_path, video_path, output_path
        ]
        for p in path_args:
            if p is not None:
                p = os.path.abspath(p)

        self.net = Darknet(model_config, device=self.device)
        self.net.load_weights(weight_path)
        self.net.eval()

        self.prob_threshold = prob_threshold
        self.class_names = class_names
        self.show_fps = show_fps
        self.nms_iou_thresh = nms_iou_thresh

    def inference(self, images, resize=True
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
        net = self.net
        device = self.device
        prob_thresh = self.prob_threshold
        nms_iou_thresh = self.nms_iou_thresh

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

    def draw_boxes(self, frames, bbox_tlbr, class_idx, class_prob):
        draw_boxes(frames, bbox_tlbr, class_idx=class_idx, class_names=self.class_names, class_prob=class_prob)

    def detect_in_cam(
        self, frames=None, cam_id=2
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
        net = self.net
        device = self.device
        prob_thresh = self.prob_threshold
        class_names = self.class_names
        show_fps = self.show_fps
        nms_iou_thresh = self.nms_iou_thresh

        # Number of frames to average for computing FPS.
        num_fps_frames = 30
        previous_fps = deque(maxlen=num_fps_frames)
        cam = cv2.VideoCapture(0)
        print("CLASS NAMES: {}".format(class_names))
        while True:
            loop_start_time = time.time()
            ret_val, frame = cam.read()

            bbox_tlbr, probs, class_idx = self.inference(images=frame)[0]
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