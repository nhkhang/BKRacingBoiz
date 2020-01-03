import cv2
import numpy as np
import gmm_train
import direction
import sys
import time

LIST = [53, 76, 98, 71, 200]
PIC_ORD = 87
VID_SRC = 'public4.mp4'
TEST_SRC = 'road_pic/lane_'+ str(PIC_ORD) + '.jpg'


def main():
    cap = cv2.VideoCapture(VID_SRC)
    count = 0
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        print(count)
        if count == 10:
            count = 0
            velo, angle = direction.decision(frame)

        # ret, frame = cap.read()
        # cv2.imshow('vid', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

def test():
    for i in LIST:
        img = cv2.imread('road_pic/lane_' + str(i) + '.jpg')
        img = img[30:,:]
        start = time.time()
        obs = [False, False]
        velo, angle = direction.decision(img, obs)
        print('Angle = ' + str(angle))
        print("Return velo, angle after " + str(time.time() - start))
        print("========")
        cv2.imshow('img', img)
        cv2.waitKey(0)
        # cv2.imshow('processing', img)
        # cv2.waitKey(0)
    # img = cv2.imread(TEST_SRC)
    # img = img[30:, :]
    # direction.decision(img)

if __name__ == '__main__':
    test()