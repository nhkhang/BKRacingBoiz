import cv2
import numpy as np
import gmm_train
import direction
import sys

LIST = [138, 139, 188]
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
        direction.decision(img)
        cv2.imshow('processing', img)
        cv2.waitKey(0)
    # img = cv2.imread(TEST_SRC)
    # img = img[30:, :]
    # direction.decision(img)

if __name__ == '__main__':
    main()