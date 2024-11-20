import cv2


class Closer:
    def __init__(self, kernel_size=3, iterations=10, output_process=False):
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.output_process = output_process

    def __call__(self, image):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.kernel_size, self.kernel_size)
        )

        closed = cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            kernel
        )

        if self.output_process:cv2.imwrite('output/Closed.jpg', closed)

        return closed
