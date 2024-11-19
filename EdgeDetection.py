import argparse
import cv2

max_lowThreshold = 100
title_tracker = 'Min Threshold:'
ratio = 3
kernel_size = 3
src = 'photo_2024-11-19_12-31-26.jpg'
src_img = cv2.imread(cv2.samples.findFile(src))
src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)


def CannyThreshold(val):
    low_threshold = val
    img_blur = cv2.blur(src_gray, (5, 5))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold * ratio)
    mask = detected_edges != 0
    dst = src_img * (mask[:,:,None].astype(src_img.dtype))
    cv2.imshow('Edge Map', dst)

cv2.namedWindow('Edge Map',cv2.WINDOW_NORMAL)
cv2.createTrackbar(title_tracker,'Edge Map',0, max_lowThreshold, CannyThreshold)

CannyThreshold(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
