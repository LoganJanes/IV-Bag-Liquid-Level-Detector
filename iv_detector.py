import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import Tuple, Optional, List

class IVBagLevelDetector:
    def __init__(self, debug=False):
        self.debug = debug
        self.processing_steps = {}

#Our first step is Grayscaling the image to remove it's colour with the following function:

def step1_grayscale(self, image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    if self.debug:
        self.processing_steps['grayscale'] = gray_image
    return gray_image