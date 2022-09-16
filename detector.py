import joblib
import os
import inspect
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import copy
import scipy
import pathlib
from math import sqrt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from models.common import Conv
from models.yolo import Model
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, \
    scale_coords,scale_coords_landmarks,filter_boxes

class FaceDetector:
    def __init__(self, paramters_name='yolov5n_state_dict.pt', config_name='yolov5n.yaml', 
    gpu = 0, min_face =100, target_size = None, frontal = None):
        self._class_path = os.path.dirname(inspect.getfile(self.__class__))
        self.gpu = gpu
        self.target_size = target_size
        self.min_face = min_face
        self.frontal = frontal
        self.detector = self.init_detector(paramters_name, config_name)
    def init_detector(self, paramters_name, config_name)
        print(self.gpu)
