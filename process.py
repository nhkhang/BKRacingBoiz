import cv2
import numpy as np
import gmm_train
import preprocess
import time

LIST = [140]
PIC_ORD = 140
INPUT = cv2.imread('road_pic/lane_' + str(PIC_ORD)+'.jpg')


def cluster_gmm(img, label):
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

    return cluster_img

def morph_lane(bin_img):
    kernel = np.ones((3, 3), np.uint8)
    after_erode = cv2.erode(bin_img, kernel, iterations=3)
    after_dilate_erode = cv2.dilate(after_erode, kernel, iterations=5)
    return after_dilate_erode

def remove_backfround(img, label, black_point):
    origin = img.copy()
    img[label == label[black_point]] = 255

    _, thresh = cv2.threshold(src=img, thresh=254, maxval=255, type=cv2.THRESH_BINARY_INV)
    n_comps, black_components = cv2.connectedComponents(thresh) # return number of components and pixel-label of image
    for comp_label in range(n_comps):
        comp = np.zeros_like(thresh)
        comp[black_components == comp_label] = 1
        if cv2.countNonZero(comp) < 2000:
            thresh[black_components == comp_label] = 0

    fill_black = black_component_2white(thresh)
    # compare = np.hstack((origin, img, thresh, fill_black))
    # cv2.imshow('input | black 2 white | ', compare)
    # cv2.waitKey(0)
    return fill_black

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

def road_component(lane_morph):
    _, thresh_white = cv2.threshold(src=lane_morph, thresh=254, maxval=255, type=cv2.THRESH_BINARY_INV)
    n_comps, lane_components = cv2.connectedComponents(thresh_white)
    origin = lane_morph.copy()
    for comp in range(n_comps):
        mask = np.zeros_like(lane_morph)
        mask[lane_components == comp] = 255
        # note: adjust the number
        if cv2.countNonZero(mask) < 256:
            lane_morph = cv2.bitwise_and(lane_morph, lane_morph, mask=~mask)
    # compare = np.hstack((origin, lane_morph))
    # cv2.imshow('compare', compare)
    # cv2.waitKey(0)
    lane = black_component_2white(lane_morph)
    return lane

def detect_lane(img):
    start = time.time()
    after_process, black_point = preprocess.preprocess(img)
    label = gmm_train.predict(after_process) # ok
    # cluster = cluster_gmm(img, label)
    start_rm_bkgr = time.time()
    lane_bin = remove_backfround(after_process, label, black_point)
    print("Remove background in " + str(time.time() - start_rm_bkgr))
    morph_cluster = time.time()
    lane_morph = morph_lane(lane_bin)
    lane_cluster = road_component(lane_morph)
    print("morph and road cluster in " + str(time.time() - morph_cluster))
    print("Detect lane in " + str(time.time() - start))
    # if __name__ == '__main__':
    #     combine_step = np.hstack((after_process, lane_bin, lane_morph, lane_cluster))
    #     return combine_step

    return lane_cluster

def test_list():
    for i in LIST:
        img = cv2.imread('road_pic/lane_' + str(i)+'.jpg')
        test = detect_lane(img)
        cv2.imshow('processing', test)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_list()
