#! /usr/bin/python
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
# import rospy
# from sensor_msgs.msg import Image, CompressedImage
# from cv_bridge import CvBridge, CvBridgeError

# bridge = CvBridge()

SIGN_SIZE = 500
TURN_SPEED = 40
TURN_ANGLE = np.pi / 2
TURN_TIME = 4

turn_mode = 0
turn_duration = 0

svclassifier = pickle.load(open('/home/nguyendat/catkin_ws/src/test/src/signDetect', 'rb'))
print("fuck1")


def detect_sign(img):
    # lower_blue = np.array([70, 55, 30])
    # upper_blue = np.array([220, 140, 80])
    frame = img[0:int(len(img[0]) / 2), 0:]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    # 215 50 37 215 49 37
    lower_blue = np.array([70, 55, 30])
    upper_blue = np.array([220, 140, 80])

    # lower_blue = np.array([70, 55, 20])
    # upper_blue = np.array([220, 140, 80])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(contours)
    if contours == []:
        return None
    # list_box = []
    size = []
    box = []
    for item in contours[0]:
        tmp = cv2.boundingRect(item)
        if tmp[3] > 10 and tmp[2] > 10 and tmp[3] < 50 and tmp[2] < 50:
            box = tmp
            size = [tmp[2], tmp[3]]
            break
    if box == []:
        return size, None
    # cv2.drawContours(802921frame, contours[0], -1, (0, 255, 0), 1)
    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 1)
    # print(box)
    sign = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    sign = cv2.resize(sign, (32, 32), interpolation=cv2.INTER_AREA)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow(img_name, sign)
    # cv2.imshow("name", frame)
    # cv2.imshow("res", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return size, sign

def turn(sign_size, sign_type):
    # if still in previous turning duration, keep going
    if sign_size is None:
        return TURN_SPEED, turn.angle

    global turn_duration
    turn_duration = TURN_TIME

    if sign_type == 'right':
        turn.angle = TURN_ANGLE
    else:
        turn.angle = -TURN_ANGLE

    return TURN_SPEED, 0


def image_callback(ros_data):
    global turn_duration, turn_mode

    # if still in turning , keep it going
    if turn_duration != 0:
        return turn(None, None)

    # get image data
    np_arr = np.fromstring(ros_data.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imshow('View', frame)
    cv2.waitKey(30)

    # detect signs
    sign_size, sign = detect_sign(frame)
    if sign is not None:
        sign = svclassifier.predict(np.reshape(sign, (1, -1)))
        turn_mode = sign != 'nochange'

    if turn_mode:
        turn(sign_size, sign)
    else:
        forward()

    # other decisions

    # return image_np


# rospy.init_node('image_listener')
# image_topic = "/team1/camera/rgb/compressed"
# rospy.Subscriber(image_topic, CompressedImage, image_callback, queue_size=1)
# rospy.spin()

# if __name__== '__main_':
#     print("wtf")
#     main()
