import cv2

from MarkerChecker import OMR
import time

# start = time.time()
image = cv2.imread('output/QalamChi.jpg')
# image = cv2.imread('assets/Scan_0003.jpg')

start = time.time()
answered, image_res = OMR.start_process(image)
end = time.time()
cv2.imshow('image', image_res)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(end - start)
