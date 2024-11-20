import cv2


class EdgeDetector:
    def __init__(self,output_process=False):
        self.output_process = output_process

    def __call__(self,image,thresh1=50,thresh2=150,aperture_size=3):
        edges = cv2.Canny(image,thresh1,thresh2,apertureSize=aperture_size)
        if self.output_process:cv2.imwrite('output/edges.jpg',edges)
        return edges