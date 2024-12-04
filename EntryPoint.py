import cv2

from MarkerChecker import OMR
from processors.ContourCornersDetector import ContourCornersDetector

image = cv2.imread('output/scanned/photo_3_2024-12-03_21-33-54.jpg')
answered,image_res = OMR.start_process(image)
cv2.imshow('Original', image_res)
cv2.waitKey(0)
cv2.destroyAllWindows()

