import cv2
import numpy as np

from pre_processors.Denoiser import Denoiser
from pre_processors.HSVFilter import HSVFilter
from pre_processors.OtsuThresholder import OtsuThresholder
from pre_processors.Resizer import Resizer
from processors.HoughLineCornerDetector import HoughLineCornerDetector


class PageExtractorByHough:
    def __init__(self,pre_processors,corner_detector,output_process = False):
        assert isinstance(pre_processors,list),"pre_processors must be a list"

        self._preprocessors = pre_processors
        self._corner_detector = corner_detector
        self.output_process = output_process


    def __call__(self,image_path):
        # Read the imagee from path
        self._image = cv2.imread(image_path)
        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
        # Processed image that at first is just original image
        self._processed = self._image

        # iterate on preprocessors and execute them
        for pre_processor in self._preprocessors:
            self._processed = pre_processor(self._processed)

        # detect corners and set it to interceptions
        self._interceptions = self._corner_detector(self._processed)

        return self._extract_page()

    def _extract_page(self):
        # obtain a consistent order of the points and unpack them
        # individually
        pts = np.array([
            (x,y)
            for interception in self._interceptions
            for x,y in interception
        ])

        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordinates or the top-right and top-left x-coordinates
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],  # Top left point
            [max_width - 1, 0],  # Top right point
            [max_width - 1, max_height - 1],  # Bottom right point
            [0, max_height - 1]],  # Bottom left point
            dtype="float32"  # Date type
        )

        # compute the perspective transform matrix and then apply it
        m = cv2.getPerspectiveTransform(rect,dst)
        wrapped = cv2.warpPerspective(self._processed, m, (max_width,max_height))

        return wrapped

    def _order_points(self, pts):
        # initialize a list of coordinates that will be ordered such that
        # 1st point -> Top left
        # 2nd point -> Top right
        # 3rd point -> Bottom right
        # 4th point -> Bottom left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect



if __name__ == "__main__":
    page_extractor = PageExtractorByHough(
        pre_processors = [
            HSVFilter(output_process=True),
            Resizer(height = 1280, output_process = True),
            Denoiser(strength = 8, output_process = True),
            OtsuThresholder(output_process = True)
        ],
        corner_detector = HoughLineCornerDetector(
            rho_acc=1,
            theta_acc=180,
            thresh=100,
            output_process = True
        )
    )
    # extracted = page_extractor('assets/photo_2024-11-20_12-47-14.jpg')
    extracted = page_extractor('assets/photo_2024-11-20_17-16-00.jpg')
    # extracted = page_extractor('assets/QalamChi.jpg')
    denoise = Denoiser()
    ex = denoise(extracted)
    cv2.imshow('denoised',ex)

    cv2.imwrite("output/output.jpg", extracted)
    cv2.imshow("Extracted page", extracted)
    cv2.waitKey(0)
