import cv2
import numpy as np
import gmm_train
import direction
import sys

VID_SRC = 'public4.mp4'
TEST_SRC = 'road_pic/lane_1.jpg'


def main():
    cap = cv2.VideoCapture(VID_SRC)
    while cap.isOpened():
        ret, frame = cap.read()
        velo, angle = direction.decision(frame)

def test():
    img = cv2.imread(TEST_SRC)
    img = img[30:, :]
    direction.decision(img)

if __name__ == '__main__':
    main()
