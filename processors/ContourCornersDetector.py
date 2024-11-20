import numpy as np
import cv2

from pre_processors.Closer import Closer
from pre_processors.EdgeDetector import EdgeDetector
from pre_processors.GrabCut import GrabCut
from pre_processors.Resizer import Resizer


class ContourCornersDetector:

    def __call__(self,image):
        self.image = image
        self.original_image = image

        # Step 1 : If it's big , make it small
        dim_limit = 1080
        max_dim = max(image.shape)
        if max_dim > dim_limit:
            resizer = Resizer(dim_limit,True)
            self.image = resizer(image)

        # Step 2 : Perform Morphology
        closer = Closer(output_process=True)
        self.image = closer(self.image)

        # Step 3: Grab Cut
        grab_cut = GrabCut(output_process=True)
        self.image = grab_cut(image = self.image)

        # Step 4: Edge Detect
        edge = EdgeDetector(True)
        self.image = edge(self.image)

        # Step 5: Detecting Contour
        self.image = self.detect_biggest_contour(self.image)

        # Step 6: Finding Corners
        corners = self.calc_contour_corners(self.image)

        # Step 7: Destination Corners

        destination_corners = self.find_dst(corners)

        h, w = self.original_image.shape[:2]

        # Getting the homography.
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

        # Perspective transform using homography.
        final = cv2.warpPerspective(self.original_image, M, (destination_corners[2][0], destination_corners[2][1]),
                                    flags=cv2.INTER_LINEAR)

        return final

    def find_dst(self,points):
        (tl, tr, br, bl) = points
        # Finding the maximum width.
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Finding the maximum height.
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Final destination co-ordinates.
        destination_corners = [[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]]

        return self.order_points(destination_corners)

    def order_points(self, points):
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(points)
        s = pts.sum(axis=1)

        # Top-left point will have the smallest sum.
        rect[0] = pts[np.argmin(s)]

        # Bottom-right point will have the largest sum.
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)

        # Top-right point will have the smallest difference.
        rect[1] = pts[np.argmin(diff)]

        # Bottom-left will have the largest difference.
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect.astype('int').tolist()

    def detect_biggest_contour(self, canny_image):

        # Finding contours for the detected edges.
        contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Keeping only the largest detected contour.
        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        return page

    def calc_contour_corners(self,page):

        # Detecting Edges through Contour approximation.
        # Loop over the contours.
        corners = []
        if len(page) == 0:
            return self.original_image

        for c in page:
            # Approximate the contour.
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            # If our approximated contour has four points.
            if len(corners) == 4:
                break

        # Sorting the corners and converting them to desired shape.
        corners = sorted(np.concatenate(corners).tolist())

        corners = self.order_points(corners)

        return corners