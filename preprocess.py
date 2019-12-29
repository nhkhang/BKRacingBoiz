import cv2
import numpy as np
import time
from sklearn.mixture import GaussianMixture


def oversaturated_filter(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([180, 35, 255]))
    hsv_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB), (i, j)

def resize(img, type):
    height, width = img.shape[:2]

    if type == 'dataset':
        img = img[30:, :]

    crop = img[int(height/3):, :]
    return crop

def data_preprocess(file_name):
    res = cv2.imread(file_name)
    res = resize(res, 'dataset')
    res, _ = oversaturated_filter(res)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

    mask = np.zeros(res.shape, dtype=np.uint8)
    x, y = res.shape[0]//3, res.shape[1]//3
    mask[x: x*2, y: y*2] = 1
    min_, _, _, _ = cv2.minMaxLoc(res-90, mask=mask)
    res += np.uint(110 - int(min_+90))
    return res

def preprocess(img):
    res = resize(img, 'none')
    start = time.time()
    res, black_point = oversaturated_filter(res)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    print("Preprocessing in " + str(time.time() - start))
    return res, black_point