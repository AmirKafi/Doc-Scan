import cv2
import numpy as np

class GrabCut:
    def __init__(self,output_process = False):
        self.output_process = output_process

    def __call__(self,image):
        mask = np.zeros(image.shape[:2], np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        rect = (20, 20, image.shape[1] - 20, image.shape[0] - 20)
        cv2.grabCut(image, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        grab_cut_img = image * mask2[:, :, np.newaxis]

        if self.output_process: cv2.imwrite('assets/GrabCut.jpg', grab_cut_img)

        return grab_cut_img