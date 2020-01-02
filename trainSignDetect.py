import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def detect_sign(img):
    # lower_blue = np.array([70, 55, 30])
    # upper_blue = np.array([220, 140, 80])
    frame = img[0:int(len(img[0])/2), 0:]
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


def svm_train(input_d, label):
    x_train, x_test, y_train, y_test = train_test_split(input_d, label, test_size=0.20)
    svclassifier = SVC(kernel='linear', C = 0.5)
    svclassifier.fit(x_train, y_train)
    y_pred = svclassifier.predict(x_test)
    y_pred = y_pred.reshape((len(y_pred), 1))
    y_test = y_test.to_numpy().reshape((len(y_test), 1))
    y = np.hstack((y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return svclassifier

if __name__ == "__main__":
    folder = ['Map1Data', 'Map2Data', 'Map3Data']
    path = '/home/nguyendat/PycharmProjects/SignRecognition/'
    list_input = []
    count = 0
    labels = pd.read_csv("/home/nguyendat/PycharmProjects/SignRecognition/labels.csv")
    labels = labels['class']
    # Traverse 3 folder of Data
    for i in range(0, 617):
        im_name = "/home/nguyendat/PycharmProjects/SignRecognition/Map1Data/Sign"
        im_name = im_name + str(i) + ".png"
        frame = cv2.imread(im_name)
        size, sign_detected = detect_sign(frame)
        list_input.append(sign_detected)
        count += 1
    for i in range(0, 236):
        im_name = "/home/nguyendat/PycharmProjects/SignRecognition/Map2Data/Sign"
        im_name = im_name + str(i) + ".png"
        frame = cv2.imread(im_name)
        size, sign_detected = detect_sign(frame)
        list_input.append(sign_detected)
        count += 1
    for i in range(0, 366):
        im_name = "/home/nguyendat/PycharmProjects/SignRecognition/Map3Data/Sign"
        im_name = im_name + str(i) + ".png"
        frame = cv2.imread(im_name)
        size, sign_detected = detect_sign(frame)
        list_input.append(sign_detected)
        count += 1
    data_images = np.reshape(list_input, (count, -1))
    svclassifier = svm_train(data_images, labels)
    pickle.dump(svclassifier, open('signDetect', 'wb'))
    