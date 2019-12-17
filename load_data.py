import numpy as np
import cv2
import os


def load_images():
    num_pics = 9
    lane_images = []

    for i in range(num_pics):
        img = cv2.imread("road_pic/lane_" + i.__str__() + ".jpg")
        lane_images.append(img)

    return lane_images
