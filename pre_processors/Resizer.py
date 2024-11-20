import cv2

class Resizer:
    """Resizes image"""

    def __init__(self, height=1280, output_process=False):
        self._height = height
        self.output_process = output_process

    def __call__(self, image):
        if image.shape[0] <= self._height: return image
        ratio = round(self._height / image.shape[0], 3)
        width = int(image.shape[1] * ratio)
        dim = (width, self._height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        if self.output_process: cv2.imwrite('output/resized.jpg', resized)
        return resized
