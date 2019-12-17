import cv2
import imageio
import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.patches import Ellipse
from PIL import Image
from sklearn import datasets
from skimage import measure
from sklearn.mixture import GaussianMixture
import direction


def oversaturated_filter(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([180, 35, 255]))
    hsv_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 0:
                return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB), (i, j)


def resize(img):
    y1 = 30
    crop = img[y1:, :]
    return get_crop_img(crop)


def get_crop_img(img):
    height, width = img.shape[:2]
    crop_img = img[int(height/3):, :]
    return crop_img


def resize_pixels_img(img):
    img = resize(img)
    nPx = img.shape[0] * img.shape[1]
    pixels = np.zeros((nPx, 5), dtype=float)

    j = 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            pixels[j] = [row, col, img[row, col, 0], img[row, col, 1], img[row, col, 2]]
            pixels[j] /= [17, 17, 15, 15, 15]
            j = j + 1
    return pixels


def load_images():
    num_pics = 189
    lane_images = []

    for i in range(1, num_pics):
        img = img_preprocess("road_pic/lane_" + str(i) + ".jpg")
        lane_images.append(img)

    pixels = np.reshape(lane_images, (-1, 1))

    # for pix in pixels:
    #     pix[0] >>= 1
    #     pix[2] >>= 1

    return pixels


def get_GMM_model(n_clusters):
    # X = load_images()
    return GaussianMixture(n_components=n_clusters, covariance_type='diag').fit(load_images())


def get_target_label(gmm_model, n_labels, trusted_file):
    trusted_road_img = cv2.imread(trusted_file)

    # trusted_road_img = cv2.cvtColor(trusted_road_img, cv2.COLOR_BGR2HSV)
    trusted_road_img = cv2.cvtColor(trusted_road_img, cv2.COLOR_BGR2GRAY)
    trusted_road_img = trusted_road_img[200:270, 100:200]

    cv2.imshow('trusted NOT', trusted_road_img)

    trusted_road_img = np.reshape(trusted_road_img, (-1, 1))
    trusted_road_img = gmm_model.predict(trusted_road_img)

    label_cnt = np.zeros(n_labels, dtype=int)
    for pix in trusted_road_img:
        label_cnt[trusted_road_img[pix]] += 1

    max_label = 0
    for i in range(1, n_labels):
        if label_cnt[i] > label_cnt[max_label]:
            max_label = i

    return max_label


def predict(input, gmm_model):
    pixels = np.reshape(input, (-1, 1))
    label_input = gmm_model.predict(pixels)
    label_input = np.reshape(label_input, (input.shape[0], input.shape[1]))

    return label_input


def color_lane(img, label_img, lane_label):
    lane_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    lane_img[label_img == lane_label] = 255

    return lane_img


def print_clusters(label, img):
    color = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 255),
              (0, 0, 0),
             (255, 255, 0),
             (255, 0, 255),
             (0, 255, 255)]

    cluster_img = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            cluster_img[i][j] = color[label[i][j]]

    print_clusters.counter += 1
    cv2.imshow('cluster ' + str(print_clusters.counter), cluster_img)
    return cluster_img


def morph_lane(bin_img):
    kernel = np.ones((3, 3), np.uint8)

    bin_img = cv2.erode(bin_img, kernel, iterations=3)
    bin_img = cv2.dilate(bin_img, kernel, iterations=5)

    # cv2.imshow('morph' + str(morph.counter), bin_img)

    return bin_img


def morph_snow(bin_img):
    kernel = np.ones((3, 3), np.uint8)

    bin_img = cv2.erode(bin_img, kernel, iterations=2)
    bin_img = cv2.dilate(bin_img, kernel, iterations=10)
    return bin_img


def img_preprocess(file_name):
    res = cv2.imread(file_name)
    res = resize(res)
    res, _ = oversaturated_filter(res)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

    mask = np.zeros(res.shape, dtype=np.uint8)
    x, y = res.shape[0]//3, res.shape[1]//3
    mask[x: x*2, y: y*2] = 1
    min_, _, _, _ = cv2.minMaxLoc(res-90, mask=mask)
    res += np.uint(110 - int(min_+90))
    return res


def preprocess(img):
    res = get_crop_img(img)
    res, black_point = oversaturated_filter(res)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)

    mask = np.zeros(res.shape, dtype=np.uint8)
    x, y = res.shape[0] // 3, res.shape[1] // 3
    mask[x: x * 2, y: y * 2] = 1
    min_, _, _, _ = cv2.minMaxLoc(res - 90, mask=mask)
    res += np.uint(110 - int(min_ + 90))
    return res, black_point


def roi_snow(img_lane, origin):
    nRow, nCol = img_lane.shape[:2]
    vertices = []

    stop = 0
    for j in range(1, nCol - 1):
        for i in range(1, nRow - 1):
            if img_lane[i, j] != 0:
                vertices.append([j, i])
                stop = 1
                break
        if stop:
            break

    stop = 0
    for j in range(1, nCol - 1):
        for i in range(1, nRow - 1):
            if img_lane[i, nCol - j] != 0:
                vertices.append([nCol - j, i])
                stop = 1
                break
        if stop:
            break

    vertices.append([nCol - 1, nRow - 30])
    vertices.append([nCol - 1, nRow - 1])
    vertices.append([0, nRow-1])
    vertices.append([0, nRow - 30])

    print(vertices)

    mask = np.zeros_like(origin)
    cv2.fillConvexPoly(mask, np.array(vertices), 255)
    return mask


def road_component(lane_after_morph):
    n_comps, lane_components = cv2.connectedComponents(lane_after_morph)
    for comp in range(n_comps):
        mask = np.zeros_like(lane_after_morph)
        mask[lane_components == comp] = 255
        # note: adjust the number
        if cv2.countNonZero(mask) < 256:
            lane_after_morph = cv2.bitwise_and(lane_after_morph, lane_after_morph, mask=~mask)
    lane_after_morph = black_component_2white(lane_after_morph)
    return lane_after_morph


def black_component_2white(img):
    _, thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)
    n_comps, black_components = cv2.connectedComponents(thresh)
    for comp in range(n_comps):
        mask = np.zeros_like(thresh)
        mask[black_components == comp] = 255
        if cv2.countNonZero(mask) > 300:
            thresh = cv2.bitwise_and(thresh, thresh, mask=~mask)
    thresh = cv2.bitwise_or(img, thresh)
    return thresh


def not_black(img, label, black_point):
    img[label == label[black_point]] = 255
    _, thresh = cv2.threshold(src=img, thresh=254, maxval=255, type=cv2.THRESH_BINARY_INV)

    n_comps, black_components = cv2.connectedComponents(thresh)
    for comp_label in range(n_comps):
        comp = np.zeros_like(thresh)
        comp[black_components == comp_label] = 1
        if cv2.countNonZero(thresh) < 2000:
            thresh[black_components == comp_label] = 0

    return thresh


def all(img, gmm, road_label, snow_label):
    after_process, black_point = preprocess(img)
    label = predict(after_process, gmm)
    cluster = print_clusters(label, after_process)
    # lane_detect_bin = color_lane(after_process, label, road_label)
    # snow_bin = color_lane(after_process, label, snow_label)

    lane_detect_bin = not_black(after_process, label, black_point)
    lane_after_morph = morph_lane(lane_detect_bin)

    # road component
    lane_after_morph = road_component(lane_after_morph)
    cv2.imshow('lane_morph', lane_after_morph)

    # roi_of_snow = roi_snow(lane_after_morph, after_process)
    # road_snow = cv2.bitwise_or(lane_after_morph, roi_of_snow)

    # a_road = cv2.bitwise_and(after_process, road_snow)
    final = np.hstack((after_process, lane_detect_bin, lane_after_morph))
    return lane_after_morph


