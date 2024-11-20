import cv2
import numpy as np

from processors.ContourCornersDetector import ContourCornersDetector

contour_page_extractor = ContourCornersDetector()

img = cv2.imread('assets/QalamChi.jpg')
final = contour_page_extractor(img)
cv2.imshow('original', img)
cv2.imshow('final', final)
cv2.waitKey(0)
cv2.destroyAllWindows()