import cv2

from MarkerChecker import OMR
import time

# start = time.time()
# image = cv2.imread('output/QalamChi.jpg')
# image = cv2.imread('assets/Scan_0011.jpg')

start = time.time()
image = cv2.imread('output/QalamChi.jpg')
answered, image_res = OMR.start_process(image)
end = time.time()
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image', image_res)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(end - start)
