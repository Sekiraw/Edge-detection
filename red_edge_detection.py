import cv2
import numpy as np


class Beadando:
    def __init__(self):
        self.a = None
        self.img = cv2.imread('car_numberplate_rs.jpg', cv2.IMREAD_COLOR)
        self.normalized = None
        self.im_thresh = None
        self.img_np_edge = None
        self.magn_thresh = 0.2
        self.ksize = 7
        self.imgLab = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)

    def normalize(self, name, img, show_img=True):
        self.normalized = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        self.a = cv2.medianBlur(img, self.ksize)
        if show_img:
            cv2.imshow(name, self.normalized)
            cv2.imshow("a", self.a)

    def blur(self, name, img, show_img=True):
        noise = np.zeros(img.shape, np.int16)
        imnoise = cv2.add(img, noise, dtype=cv2.CV_8UC1)
        self.imnoisegauss5x5 = cv2.GaussianBlur(imnoise, (5, 5), sigmaX=2.0, sigmaY=2.0)
        if show_img:
            cv2.imshow(name, self.imnoisegauss5x5)

    def split_n_merge(self, name, img, show_img=True):
        self.null, self.green, self.blue = cv2.split(img)
        lower = 150
        upper = 256
        self.blue[self.blue > upper] = upper
        self.blue[self.blue < lower] = lower
        im = cv2.normalize(self.blue, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        self.green[self.green > upper] = upper
        self.green[self.green < lower] = lower
        self.merged = cv2.normalize(self.green, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        if show_img:
            cv2.imshow(name, self.merged)

    def segment(self, name, img, show_img=True):
        self.im_thresh = np.ndarray(img.shape, img.dtype)
        self.im_thresh.fill(0)
        self.im_thresh[(img >= 1) & (img <= 255)] = 255
        if show_img:
            cv2.imshow(name, self.im_thresh)

    def edge(self, name, img, show_img=True):
        self.edges = cv2.Canny(img, 100, 200, None, 5, True)
        if show_img:
            cv2.imshow(name, self.edges)

    def korvonal(self, name, img, show_img=True):
        self.img_np_edge = self.img.copy()
        self.img_np_edge[img > 0] = [0, 0, 255]
        if show_img:
            cv2.imshow(name, self.img_np_edge)

    def save_result(self, name, file):
        cv2.imwrite(name, file)

    def run(self):
        # cv2.imshow("base", self.img)
        self.normalize("normalized", self.img)
        self.blur("gauss", self.a)
        self.split_n_merge("merged", self.imgLab, False)
        # self.segment("segment", self.merged)
        self.segment("segment2", self.merged, False)
        self.edge("edge", self.im_thresh)
        self.korvonal("korvonal", self.edges)
        self.save_result("result.png", self.img_np_edge)

        cv2.waitKey(0)


if __name__ == '__main__':
    beadando = Beadando()
    beadando.run()
