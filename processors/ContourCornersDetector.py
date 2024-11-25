import cv2
import numpy as np

from pre_processors.EdgeDetector import EdgeDetector
from pre_processors.GrabCut import GrabCut
from pre_processors.Resizer import Resizer


class ContourCornersDetector:

    def __call__(self, image):

        # Step 1 : If it's big , make it small for better performance
        dim_limit = 1080
        max_dim = max(image.shape)
        if max_dim > dim_limit:
            resizer = Resizer(dim_limit, True)
            image = resizer(image)

        # Make a copy and send the copy through process
        self.org_img = image.copy()
        processed_img = image

        # Step 3: Grab Cut
        grab_cut = GrabCut(output_process=True)
        processed_img = grab_cut(processed_img)

        # Step 4: Edge Detect
        edge = EdgeDetector(True)
        canny = edge(processed_img)

        # Step 5: Detecting Contour
        # Gets the canny image to get contour
        page = find_largest_contour(canny)

        # Step 6: Finding Corners
        # Gets extracted contour to get its corners
        corners = self.get_corners(page)

        # Make the points to be in order
        ordered = order_points(corners)

        # Final Step : Turn it to Bird's eye view
        final = perspective_transform(self.org_img, ordered)

        return final

    def get_corners(self, page):

        corners = approximate_corners(page)

        missing_side, corners = detect_missing_corner(corners)

        copy = self.org_img.copy()
        order = order_corners(corners)
        final_image, filtered_intersections = draw_lines_find_intersections_with_missing_side(
            copy, order, missing_side
        )

        return filtered_intersections

def find_largest_contour(canny_image):
    contours, _ = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def order_points(points):
    points = np.array(points)
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect.astype("int").tolist()

def crop_region(image, corners):
    """
    Crops the region defined by the corners from the image.

    Args:
        image (ndarray): Original image.
        corners (list): List of four intersections (x, y).

    Returns:
        ndarray: Cropped image.
    """
    # Ensure corners are sorted (top-left, top-right, bottom-right, bottom-left)
    rect = np.array(corners, dtype="float32")

    # Compute the width and height of the new image
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    # Define the destination points for the perspective transformation
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(rect, dst)

    # Perform the perspective warp
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped

def approximate_corners(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, epsilon, True)
    return corners

def detect_missing_corner(corners):
    if len(corners) == 5:
        # If there are 5 corners, we need to determine which side is missing
        # Sort corners by their positions to determine side orientation
        corners = sorted(corners, key=lambda x: (x[0][0], x[0][1]))  # Sort by x then y

        # Check distances between corners to group them into 4 sides
        distances = []
        for i in range(len(corners)):
            for j in range(i + 1, len(corners)):
                dist = np.linalg.norm(corners[i][0] - corners[j][0])
                distances.append(dist)

        # Find the missing side based on unequal distances (one side will have only one corner)
        missing_side = None
        threshold = 100  # Adjust as necessary
        for i, corner in enumerate(corners):
            if any(np.linalg.norm(corner[0] - other_corner[0]) > threshold for other_corner in corners):
                missing_side = i
                break

        return missing_side, corners

    return None, corners

def order_corners(corners):
    points = [corner[0] for corner in corners]
    points = np.array(points)

    # Calculate the center of the points
    center = np.mean(points, axis=0)

    # Sort points based on their angle relative to the center
    def angle_from_center(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])

    points = sorted(points, key=angle_from_center)

    # Return sorted corners as np.array for easier drawing
    return np.array(points, dtype=np.int32)

def calculate_intersection(line1, line2):
    if line1[0] == "vertical" and line2[0] == "vertical":
        return None  # Parallel vertical lines
    if line1[0] == "vertical":
        x = line1[1]
        y = line2[1] * x + line2[2]  # y = m*x + c for line2
        return x, y
    if line2[0] == "vertical":
        x = line2[1]
        y = line1[1] * x + line1[2]  # y = m*x + c for line1
        return x, y

    # Both are non-vertical lines: m1*x + c1 = m2*x + c2
    m1, c1 = line1[1], line1[2]
    m2, c2 = line2[1], line2[2]

    if m1 == m2:
        return None  # Parallel lines

    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

def draw_extended_lines_and_find_intersections(image, corners):
    height, width, _ = image.shape
    extended_image = image.copy()
    lines = []

    for i in range(len(corners)):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % len(corners)])  # Wrap around to connect last corner to the first

        # Calculate the line equation (y = mx + c)
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        if dx == 0:  # Vertical line
            x = pt1[0]
            lines.append(("vertical", x, None))
            cv2.line(extended_image, (x, 0), (x, height), (255, 0, 0), 2)
        else:
            slope = dy / dx
            intercept = pt1[1] - slope * pt1[0]
            lines.append(("line", slope, intercept))

            # Extend the line to the image boundaries
            x1, y1 = 0, int(intercept)  # Left edge
            x2, y2 = width, int(slope * width + intercept)  # Right edge

            cv2.line(extended_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Calculate intersections
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = calculate_intersection(lines[i], lines[j])
            if intersection:
                x, y = intersection
                if 0 <= x <= width and 0 <= y <= height:  # Ensure within image boundaries
                    intersections.append((int(x), int(y)))

    # Draw intersections on the image
    for x, y in intersections:
        cv2.circle(extended_image, (x, y), 10, (0, 255, 0), -1)  # Green dots for intersections

    return extended_image, intersections

def exclude_missing_side_intersections(intersections, missing_side_index):
    valid_intersections = []

    for i, point in enumerate(intersections):
        if i not in [missing_side_index, (missing_side_index + 1) % len(intersections)]:
            valid_intersections.append(point)

    return valid_intersections

def draw_lines_find_intersections_with_missing_side(image, corners, missing_side_index):
    height, width, _ = image.shape
    extended_image = image.copy()
    lines = []

    for i in range(len(corners)):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % len(corners)])  # Wrap around to connect last corner to the first

        # Calculate the line equation (y = mx + c)
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        if dx == 0:  # Vertical line
            x = pt1[0]
            lines.append(("vertical", x, None))
            cv2.line(extended_image, (x, 0), (x, height), (255, 0, 0), 2)
        else:
            slope = dy / dx
            intercept = pt1[1] - slope * pt1[0]
            lines.append(("line", slope, intercept))

            # Extend the line to the image boundaries
            x1, y1 = 0, int(intercept)  # Left edge
            x2, y2 = width, int(slope * width + intercept)  # Right edge

            cv2.line(extended_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Calculate intersections
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = calculate_intersection(lines[i], lines[j])
            if intersection:
                x, y = intersection
                if 0 <= x <= width and 0 <= y <= height:  # Ensure within image boundaries
                    intersections.append((int(x), int(y)))

    if missing_side_index is not None:
        # Exclude intersections along the missing side
        filtered_intersections = exclude_missing_side_intersections(intersections, missing_side_index)

        # Draw filtered intersections on the image
        for x, y in filtered_intersections:
            cv2.circle(extended_image, (x, y), 10, (0, 255, 0), -1)  # Green dots for intersections

        return extended_image, filtered_intersections

    return extended_image, intersections

def perspective_transform(image, src_points):
    """
    Perform a perspective transform on the image using the source points and destination points.

    :param image: Input image to be transformed.
    :param src_points: A tuple of 4 points in the form ((x1, y1), (x2, y2), (x3, y3), (x4, y4)).
    :return: Transformed image.
    """
    # Ensure the points are in the correct order: top-left, top-right, bottom-right, bottom-left
    src_points = np.array(src_points, dtype=np.float32)

    # Define the destination points (for a rectangular transform)
    width = max(np.linalg.norm(src_points[2] - src_points[3]), np.linalg.norm(src_points[1] - src_points[0]))
    height = max(np.linalg.norm(src_points[1] - src_points[2]), np.linalg.norm(src_points[0] - src_points[3]))

    dst_points = np.array([
        [0, 0],  # top-left
        [width - 1, 0],  # top-right
        [width - 1, height - 1],  # bottom-right
        [0, height - 1]  # bottom-left
    ], dtype=np.float32)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transform
    transformed_image = cv2.warpPerspective(image, matrix, (int(width), int(height)))

    return transformed_image