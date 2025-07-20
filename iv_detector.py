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


#Our fourth step is to introduce a region of interest so the algorithm will focus only on the liquid and not on labels, text, and other things.

def region_of_interest(self, image: np.ndarray, 
                             roi_coords: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    if roi_coords is None:
        x = width // 4
        y = height // 6
        w = width // 2
        h = 2 * height // 3
        roi_coords = (x, y, w, h)
    x, y, w, h = roi_coords
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    roi_image = cv2.bitwise_and(image, image, mask=mask)
    return roi_image, roi_coords

#Our fifth step is to use thresholding to binarize the image intensities to either white or gray. White pixels should be liquid while remaining black pixels are not.
#_ is used to ignore the threshold value and only return the resulting image.

def thresholding(self, image: np.ndarray, threshold_value: int = 127) -> np.ndarray:
    _, thresh = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

    #Our sixth step is to use morphological closing to fill in any holes in the thresholded binary image.

def morphological_closing(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return closed