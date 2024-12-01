import cv2

from MarkerChecker import OMR

image = cv2.imread('output/QalamChi.jpg')
# image = cv2.imread('assets/Scan_0011.jpg')


answered,image_res = OMR.start_process(image)

cv2.imshow('image',image_res)
cv2.waitKey(0)
cv2.destroyAllWindows()