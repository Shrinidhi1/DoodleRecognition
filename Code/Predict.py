import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import cv2 as cv

def main():
    classes = ["Candle", "Eye Glasses", "T-Shirt"]
    p = np.load('intermediate/Doodle.npy')
    # print(p.shape)
    p = p / 255
    # p = p[130:280, 250:400]

    model = load_model("doodle_best.h5")
    y = np.resize(p, (1, 64, 64, 1))
    # print(y.shape)
    # plt.imshow(p, cmap='gray')
    # plt.show()
    cnnPredict = model.predict(y)
    # print('Prediction: ' + classes[int(np.argmax(cnnPredict[0]))] + " with " + str(round(np.max(cnnPredict[0]) * 100, 2)) + ' percent confidence')
    return cnnPredict