import sys

import cv2
import numpy as np

DELAY_CAPTION = 1500
DELAY_BLUR = 100
MAX_KERNEL_LENGTH = 31

src = None
dst = None
window_name = 'Smoothing'


def display_caption(caption):
    global dst
    dst = np.zeros(src.shape, src.dtype)
    rows, cols, _ch = src.shape
    cv2.putText(dst, caption,
               (int(cols / 4), int(rows / 2)),
               cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    return display_dst(DELAY_CAPTION)

def display_dst(delay):
    cv2.imshow(window_name, dst)
    c = cv2.waitKey(delay)
    if c >= 0 : return -1
    return 0

def main(argv):
    cv2.namedWindow(window_name)

    image_name = argv[0] if len(argv) == 1 else 'european-shorthair.jpg'

    global src

    src = cv2.imread(cv2.samples.findFile(image_name))
    if src is None:
        print("Could not load image:", image_name)
        return -1
    if display_caption('Original Image'):
        return 0

    global dst
    dst = np.copy(src)
    if display_dst(DELAY_CAPTION) != 0:
        return 0

    #Homogeneous
    if display_caption('Homogeneous Image') != 0:
        return 0

    for i in range(1,MAX_KERNEL_LENGTH,2):
        dst = cv2.blur(src, (i, i))
        if display_dst(DELAY_BLUR) != 0:
            return 0

    #Gaussian
    if display_caption('Gaussian Blur') != 0:
        return 0

    for i in range(1,MAX_KERNEL_LENGTH,2):
        dst = cv2.GaussianBlur(src, (i, i), 0)
        if display_dst(DELAY_BLUR) != 0:
            return 0

    #Median
    if display_caption('Median Blur') != 0:
        return 0

    for i in range(1,MAX_KERNEL_LENGTH,2):
        dst = cv2.medianBlur(src, i)
        if display_dst(DELAY_BLUR) != 0:
            return 0

    #Bilateral
    if display_caption('Bilateral Filter') != 0:
        return 0

    for i in range(1,MAX_KERNEL_LENGTH,2):
        dst = cv2.bilateralFilter(src, i, i, i)
        if display_dst(DELAY_BLUR) != 0:
            return 0

    #Done
    display_caption('Done !')
    return 0

if __name__ == '__main__':
    main(sys.argv[1:])