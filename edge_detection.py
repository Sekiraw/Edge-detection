import cv2
import numpy as np
import os


class Beadando:
    def __init__(self):
        self.input_picture = None
        self.get_files()  # getting the input file, it can only be one, if there are more pictures, it chooses the
        # first it sees

        # self.input_picture = "car_numberplate_rs.jpg"  # simpler input file
        self.output_picture = "result.png"  # name of the output file
        self.outline_color = [0, 255, 0]  # green [0, 0, 255] to red [255, 0, 0] blue

        self.a = None
        self.img = cv2.imread(self.input_picture, cv2.IMREAD_COLOR)
        self.normalized = None
        self.im_thresh = None
        self.img_np_edge = None
        self.magn_thresh = 0.2
        self.ksize = 7
        self.imgLab = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)

    def get_files(self):
        files = os.listdir('.')
        for file in files:
            if file != ".git" and file != "README.md" and file != "result.png" and file != os.path.basename(__file__):
                self.input_picture = file

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
            pass
            #cv2.imshow(name, self.edges)

    def korvonal(self, name, img, show_img=True):
        self.img_np_edge = self.img.copy()
        self.img_np_edge[img > 0] = self.outline_color
        if show_img:
            cv2.imshow(name, self.img_np_edge)
            cv2.waitKey(0)

    def save_result(self, name, file):
        cv2.imwrite(name, file)

    def run(self):
        # cv2.imshow("base", self.img)
        self.normalize("normalized", self.img, False)
        self.blur("gauss", self.a, False)
        self.split_n_merge("merged", self.imgLab, False)
        # self.segment("segment", self.merged)
        self.segment("segment2", self.merged)
        self.edge("edge", self.im_thresh)
        self.korvonal("result", self.im_thresh)
        self.save_result(self.output_picture, self.img_np_edge)

        cv2.waitKey(0)


if __name__ == '__main__':
    beadando = Beadando()
    beadando.run()
