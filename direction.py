import cv2
import numpy as np
import process


def cur_pos(img):
    height, width = img.shape[:2]
    count_black = 0
    count_road = 0
    angle = 0
    count_black_left = 0

    for i in range(width):
        if img[height-1][width - i - 1] != 0:
            count_black = i
            break

    for i in range(width):
        if img[height - 1][i] != 0:
            count_black_left = i
            break

    if count_black > 50 and count_black_left == 0:
        # right
        angle += 0.45

    elif count_black == 0 and count_black_left > 50:
        # maybe left
        # for j in range(height):
        #     if img[height - 1 - j][width-1] == 0:
        #         count_road = j
        #         break
        # if count_road > 70:
        #     # left
        angle += -0.45

    velo = 60
    return velo, angle


def detect_lane_contour(img):
    height, width = img.shape[:2]
    padding = []
    for i in range(height):
        padding.append([0])

    img_padding = np.hstack((padding, img, padding))

    right_lane = [[height - 10, width - 1]]
    left_lane = [[height - 10, 1]]

    # right and left lane
    while left_lane[-1][0] and left_lane[-1][1] < right_lane[-1][1]:
        yy = right_lane[-1][0]
        while right_lane[-1][1] != left_lane[-1][1] and right_lane[-1][0] == yy:
            right_lane.append(choose_with_kernel(img_padding, "right", right_lane[-1][0], right_lane[-1][1]))
        while left_lane[-1][1] != right_lane[-1][1] and left_lane[-1][0] == yy:
            left_lane.append(choose_with_kernel(img_padding, "left", left_lane[-1][0], left_lane[-1][1]))

    return left_lane, right_lane


def choose_with_kernel(img, position, y, x):
    if position == "right":
        if img[y - 1][x + 1]:
            return y - 1, x + 1
        if img[y - 1][x]:
            return y - 1, x
        if img[y - 1][x - 1]:
            return y - 1, x - 1
        return y, x - 1

    if position == "left":
        if img[y - 1][x - 1]:
            return y - 1, x - 1
        if img[y - 1][x]:
            return y - 1, x
        if img[y - 1][x + 1]:
            return y - 1, x + 1
        return y, x + 1


def draw_contour(img, lane):
    left, right = detect_lane_contour(lane)
    height, width = img.shape[:2]

    for i in range(len(left) - 1):
        img[left[i][0] + int(height/3), left[i][1]] = (255, 0, 0)
    for i in range(len(right) - 1):
        img[right[i][0] + int(height/3), right[i][1]-1] = (0, 255, 0)

    obs_left, obs_right = detect_obs(left, right, img)
    for i in obs_left:
        img[left[i][0] + int(height/3), left[i][1]] = (255, 0, 255)
    for i in obs_right:
        img[right[i][0] + int(height/3), right[i][1] - 1] = (0, 255, 255)

    cv2.imshow('lane', lane)
    cv2.imshow("contour", img)
    cv2.waitKey(0)


def detect_obs(left, right, img):
    dleft = []
    dright = []
    height, width = img.shape[:2]
    n = len(left)
    obstacle_arr = []

    for i in range(0, n):
        if n < i + 21:
            break
        dy = 0
        dx = 0
        for j in range(1, 21):
            dy += (left[i][0] - left[i + j][0]) / j
            dx += (left[i + j][1] - left[i][1]) / j

        dleft.append([dy / 20, dx / 20])

        if (4 * dy) < dx:
            print('obstacles with slope of ' + str(dy/(dx+0.1)) + ' at ' + str(left[i]))
            obstacle_arr.append(i)

    n = len(right)
    obs_right = []
    for i in range(0, n):
        if n < i + 21:
            break
        dy = 0
        dx = 0
        for j in range(1, 21):
            dy += (right[i][0] - right[i + j][0]) / j
            dx += (right[i][1] - right[i + j][1]) / j

        dright.append([dy / 20, dx / 20])

        if (4 * dy) < dx:
            print('obstacles with slope of ' + str(dy / (dx + 0.1)) + ' at ' + str(right[i]))
            obs_right.append(i)

    return obstacle_arr, obs_right


def decision(img):
    lane = process.detect_lane(img)
    draw_contour(img, lane)
    velo, angle = cur_pos(lane)
    return velo, angle