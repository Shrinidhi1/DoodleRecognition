import numpy as np
import matplotlib.pyplot as plt
import cv2

num = 122

def convertColors(x):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # Invert Colors
    for i in range(150):
        for j in range(150):
            x[i, j] = 255 - x[i, j]
    return x


def cornerCorrection(x):
    for i in range(150):
        for j in range(150):
            if i==0 or i==1 or i==148 or i==149 or j == 0 or j==1 or j==148 or j==149:
                x[i, j] = 0
    return x


def increaseContrast(n):
    bias = 70
    if (n+bias)>255:
        return 255
    else:
        return n+bias


def filter(num, x):
    img = x.copy()
    for i in range(150):
        for j in range(150):
            # print(img[i, j])
            # print("\n")
            if img[i, j].any() < num:
                img[i, j] = 0
            else:
                img[i, j] = increaseContrast(img[i, j])
            # Corner correction
            img = cornerCorrection(img)
    return img


def imageMatch():
    x = np.load('intermediate/Doodle.npy').astype('float32')
    x = convertColors(x)
    img = filter(num, x)
    # plt.imshow(img)
    # plt.show()
    np.save('intermediate/Doodle.npy', img)

# imageMatch()
