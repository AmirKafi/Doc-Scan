import cv2
import numpy as np

class Closer:
    def __init__(self, kernel_size=3, iterations=10):
        self.kernel_size = kernel_size
        self.iterations = iterations

    def __call__(self, image):
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
        return closed
