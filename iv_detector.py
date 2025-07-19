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

def grayscale(self, image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    return gray_image


#Our second step is to use a median filter to  reduce salt-and-pepper noise while preserving edges of the liquid

def median_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    blurred = cv2.medianBlur(image, kernel_size)
    return blurred


#Our third step is to use canny edge detection to find the strong edges after the median filter smoothed the image and preserved edges.

def canny_edge_detection(self, image: np.ndarray, 
                               low_threshold: int = 50, 
                               high_threshold: int = 150) -> np.ndarray:
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


