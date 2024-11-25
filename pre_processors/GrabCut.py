import cv2
import numpy as np

class GrabCut:
    def __init__(self,output_process = False):
        self.output_process = output_process

    def __call__(self,img):
        rect = None
        downscale_factor = 2
        iter_count=3

        # Step 1: Resize image
        original_size = img.shape[:2]
        small_img = cv2.resize(img, (img.shape[1] // downscale_factor, img.shape[0] // downscale_factor))

        # Step 2: Define mask and models
        mask = np.zeros(small_img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Step 3: Default rectangle if none provided
        if rect is None:
            rect = (10, 10, small_img.shape[1] - 20, small_img.shape[0] - 20)

        # Step 4: Run GrabCut
        cv2.grabCut(small_img, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)

        # Step 5: Create mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented_small_img = small_img * mask2[:, :, np.newaxis]

        # Step 6: Resize back to original size
        segmented_img = cv2.resize(segmented_small_img, (original_size[1], original_size[0]))

        if self.output_process: cv2.imwrite('output/GrabCut.jpg', segmented_img)

        return segmented_img