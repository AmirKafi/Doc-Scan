import os

import cv2

from MarkerChecker import OMR
import time

from processors import OMRProcess
from processors.OMRProcess import dump_result


def get_files_in_path(path):
    files = []
    for _, _, f in os.walk(path):
        files.extend(f)
    return files



images_path = 'assets/'

image_names = get_files_in_path(images_path)
for image_name in image_names:
    print(image_name)
    image = cv2.imread(images_path + image_name)
    start_time = time.time()
    answered,image_res = OMRProcess.start_process(images_path + image_name)
    end_time = time.time()
    dump_result(answered)
    print(end_time - start_time)

