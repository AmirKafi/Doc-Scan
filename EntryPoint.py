import cv2

from processors.ContourCornersDetector import ContourCornersDetector

contour = ContourCornersDetector()

image = cv2.imread('assets/QalamChi.jpg')
result = contour(image)

cv2.imwrite('output/QalamChi.jpg', result)