import cv2
import numpy as np
import time
import preprocess
from sklearn.mixture import GaussianMixture
import pickle

N_CLUSTERS = 5
N_PICS = 448


def load_images():
    lane_images = []
    for i in range(1, N_PICS):
        img = preprocess.data_preprocess("road_pic/lane_" + str(i) + ".jpg")
        lane_images.append(img)
        print(lane_images[i-1].shape)
    pixels = np.reshape(lane_images, (-1, 1))
    return pixels

def get_GMM_model():
    X = load_images()
    return GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag').fit(X)

def store_model():
    model = get_GMM_model()
    with open('gmm_model', 'wb') as f:
        pickle.dump(model, f)

def load_model_pickle():
    with open('gmm_model', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(input):
    model = load_model_pickle()
    pixels = np.reshape(input, (-1, 1))
    label_input = model.predict(pixels)
    label_input = np.reshape(label_input, (input.shape[0], input.shape[1]))

    return label_input

if __name__ == '__main__':
    # start_train = time.time()
    # store_model()
    # training = time.time() - start_train
    # print("train time: " + str(training))
    # # test load model

    start_load_pickle = time.time()
    load_model_pickle()
    pickle_load = time.time() - start_load_pickle
    print(pickle_load)

