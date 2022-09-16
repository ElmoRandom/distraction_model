import os
import numpy as np
import torch
import cv2 as cv2

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

print("helloworld")