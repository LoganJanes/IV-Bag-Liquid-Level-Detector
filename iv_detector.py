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

    #Our seventh step is the most complex, where we will calculate the fluid height through counting the pixels within the ROI to measure fill.

    def calculate_fluid_height(self, image: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> Tuple[int, float]:
        x, y, w, h = roi_coords
        roi = image[y:y+h, x:x+w]
        fluid_top = -1
        fluid_bottom = -1
        for i in range(roi.shape[0]):
            if np.any(roi[i, :] > 0):
                fluid_top = i
                break
        for i in range(roi.shape[0] - 1, -1, -1):
            if np.any(roi[i, :] > 0):
                fluid_bottom = i
                break
        if fluid_top != -1 and fluid_bottom != -1:
            fluid_height_pixels = fluid_bottom - fluid_top
            fluid_level_percentage = (fluid_height_pixels / h) * 100
        else:
            fluid_height_pixels = 0
            fluid_level_percentage = 0.0

        return fluid_height_pixels, fluid_level_percentage

    #Our eigth step is using Contour Detection in order to detect the outlines of the liquid.

    def contour_detection(self, image: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        return filtered_contours


    #Now that we have each image processing step, we will combine them all into a full pipeline implementation.

    def detect_liquid_level(self, image_path: str, roi_coords=None, low_threshold=30):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image failed to load")
        original_image = image.copy()
        
        gray = self.grayscale(image)
        blurred = self.median_blur(gray)
        edges = self.canny_edge_detection(blurred)
        roi_edges, roi_coords = self.region_of_interest(edges, roi_coords)
        thresh = self.thresholding(roi_edges)
        closed = self.morphological_closing(thresh)
        fluid_height_pixels, fluid_level_percentage = self.calculate_fluid_height(closed, roi_coords)
        contours = self.contour_detection(closed)
        
        is_low = fluid_level_percentage < low_threshold

        if fluid_level_percentage >= 66:
             status = "HIGH"
        elif fluid_level_percentage >= 33:
            status = "MEDIUM"
        else:
             status = "LOW"

        return {
            'original_image': original_image,
            'processed_image': closed,
            'fluid_height_pixels': fluid_height_pixels,
            'fluid_level_percentage': fluid_level_percentage,
            'is_low': is_low,
            'roi_coords': roi_coords,
            'contours': contours,
            'status': status,
            'processing_steps': self.processing_steps if self.debug else None
        }
        


if __name__ == "__main__":
    detector = IVBagLevelDetector(debug=True)

    image_files = ["high_iv.jpg", "medium_iv.jpg", "low_iv.jpg"]

    for image_path in image_files:
        print(f"\n--- Processing {image_path} ---")
        results = detector.detect_liquid_level(image_path)
        print(f"Fluid Level: {results['fluid_level_percentage']:.2f}%")
        print(f"Classification: {results['status']}")


