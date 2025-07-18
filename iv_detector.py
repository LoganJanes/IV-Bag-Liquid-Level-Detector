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
