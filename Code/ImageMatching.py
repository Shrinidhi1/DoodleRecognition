import numpy as np
import matplotlib.pyplot as plt
import cv2

num = 122

# Load the doodle saved
x = np.load('intermediate/Doodle.npy').astype('float32')

# Convert to black and white
x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

# Invert Colors
for i in range(28):
    for j in range(28):
        x[i, j] = 255 - x[i, j]


def cornerCorrection(x):
    for i in range(28):
        for j in range(28):
            if i==0 or i==1 or i==26 or i==27 or j == 0 or j==1 or j==26 or j==27:
                x[i, j] = 0
    return x


def increaseContrast(n):
    bias = 70
    if (n+bias)>255:
        return 255
    else:
        return n+bias


def filter(num):
    img = x.copy()
    for i in range(28):
        for j in range(28):
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
    img = filter(num)
    # plt.imshow(img)
    # plt.show()
    np.save('intermediate/Doodle.npy', img)

# imageMatch()
