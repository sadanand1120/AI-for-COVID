import cv2
import numpy as np
import torch
from torch.autograd import Variable

from scripts.models import Darknet
from scripts.utils import load_classes, non_max_suppression_output


class AllBehaviours:
    def __init__(self):
        # Face-Mask Detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet("face-mask_model_info\\yolov3_mask.cfg", img_size=416).to(self.device)
        # Load checkpoint (ie, weights)
        self.model.load_state_dict(
            torch.load("face-mask_model_info\\yolov3_ckpt_35.pth", map_location=torch.device("cpu")))
        self.model.eval()
        self.classes = load_classes("face-mask_model_info\\mask_dataset.names")
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.mul_constant = None
        self.x = None
        self.y = None
        self.v_height = None
        self.v_width = None
        self.start_new_i_height = None
        self.start_new_i_width = None

    def faceMaskDetector(self, frame):
        if self.mul_constant is None:
            self.v_height, self.v_width = frame.shape[:2]
            # For a black image
            self.x = self.y = self.v_height if self.v_height > self.v_width else self.v_width
            # Putting original image into black image
            self.start_new_i_height = int((self.y - self.v_height) / 2)
            self.start_new_i_width = int((self.x - self.v_width) / 2)
            # For accommodate results in original frame
            self.mul_constant = self.x / 416

        org_frame = frame
        # Black image
        frame = np.zeros((self.x, self.y, 3), np.uint8)
        frame[self.start_new_i_height: (self.start_new_i_height + self.v_height),
        self.start_new_i_width: (self.start_new_i_width + self.v_width)] = org_frame
        # resizing to [416x 416]
        frame = cv2.resize(frame, (416, 416))
        # [BGR -> RGB]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # [[0...255] -> [0...1]]
        frame = np.asarray(frame) / 255
        # [[3, 416, 416] -> [416, 416, 3]]
        frame = np.transpose(frame, [2, 0, 1])
        # [[416, 416, 3] => [416, 416, 3, 1]]
        frame = np.expand_dims(frame, axis=0)
        # [np_array -> tensor]
        frame = torch.Tensor(frame)
        # [tensor -> variable]
        frame = Variable(frame.type(self.Tensor))

        # Get detections
        with torch.no_grad():
            detections = self.model(frame)
        detections = non_max_suppression_output(detections, conf_thres=0.8, nms_thres=0.3)  # [tensor(...)]
        detections = detections[0]  # tensor(...)
        final_detects = ["Face-mask"]
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # Accommodate bounding box in original frame
                x1 = int(x1 * self.mul_constant - self.start_new_i_width)
                y1 = int(y1 * self.mul_constant - self.start_new_i_height)
                x2 = int(x2 * self.mul_constant - self.start_new_i_width)
                y2 = int(y2 * self.mul_constant - self.start_new_i_height)
                total_conf = conf * cls_conf
                final_detects.append((x1, y1, x2, y2, total_conf, cls_pred))
        return final_detects
