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

    right_lane = []
    left_lane = []

    for i in range(width):
        if img_padding[height - 1][width - i] != 0:
            if i > 0:
                right_lane.append([height - 1, width - i])
            else:
                for j in range(height // 2, height):
                    if img_padding[j][width] != 0:
                        right_lane.append([j, width])
                        break
            break

    for i in range(1, width):
        if img_padding[height-1][i] != 0:
            if i > 1:
                left_lane.append([height-1, i])
            else:
                for j in range(height // 2, height):
                    if img_padding[j][1] != 0:
                        left_lane.append([j, 1])
                        break
            break

    # right and left lane
    while 8 * left_lane[-1][0] > height and left_lane[-1][1] < right_lane[-1][1]:
        yy = right_lane[-1][0]
        while right_lane[-1][1] != left_lane[-1][1] and right_lane[-1][0] >= yy:
            right_lane.append(choose_with_kernel(img_padding, "right", right_lane[-1][0], right_lane[-1][1]))
        while left_lane[-1][1] != right_lane[-1][1] and left_lane[-1][0] >= yy:
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


def detect_obs(left, right, img):
    n = len(left)
    obs_left = []

    for i in range(n-20):
        dy = 0
        for j in range(i + 1, i + 21):
            dy += (left[i][0] - left[j][0]) / ((left[j][1] - left[i][1])+0.1)

        if dy < 4:
            obs_left.append(i)

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

        if (4 * dy) < dx:
            obs_right.append(i)

    return obs_left, obs_right


def angle_calculate(right, left):
    left_grad = left[-1] - left[0]
    right_grad = right[-1] - right[0]
    grad = left_grad + right_grad

    return np.arctan(grad[1] / grad[0])


def decision(img):
    lane = process.detect_lane(img)
    draw_contour(img, lane)
    velo, angle = cur_pos(lane)
    return velo, angle
