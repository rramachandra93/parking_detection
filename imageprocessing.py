import cv2
from imutils import perspective
import numpy as np
import glob


class ImageProcess():

    def __init__(self):
        pass

    def extract_features(self, img_resize):

        global_lists = []

        for image in img_resize:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist.flatten()

            edged = cv2.Canny(image, 200, 250)
            edged = edged.flatten()

            flat = image.flatten()
            global_feature = np.hstack([hist, edged, flat])
            global_lists.append(global_feature)

        global_lists = np.asarray(global_lists)
        return global_lists

    '''rotate Images'''

    def get_rectangle(img, cooridnate_lists, WIDTH=100, HEIGHT=100):

        cropped_img_lists = []
        i = 0
        for coordinate in cooridnate_lists:
            cropped = perspective.four_point_transform(img, coordinate)
            cropped_resize = cv2.resize(cropped, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("resized- %d" % i, cropped_resize)
            cropped_img_lists.append(cropped_resize)
            i += 1
        return cropped_img_lists

    def load_image_from_path(self, path):
        img_list = []
        for files in glob.glob(path + "/*.jpg"):
            img = cv2.imread(files)
            img = self.initial_process(img)
            img_list.append(img)

        return img_list

    def initial_process(self, image, WIDTH=100, HEIGHT=100):

        img = image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        return img