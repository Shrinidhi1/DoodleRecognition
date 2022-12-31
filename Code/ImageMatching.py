import numpy as np
import matplotlib.pyplot as plt
import cv2

num = 122

# Load the doodle saved
x = np.load('intermediate/Doodle.npy')

# Convert to black and white
# x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

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
    if n+bias>255:
        return 255
    else:
        return n+bias


def filter(num):
    img = x.copy()
    for i in range(28):
        for j in range(28):
            if img[i, j] < num:
                img[i, j] = 0
            else:
                img[i, j] = increaseContrast(img[i, j])
            # Corner correction
            img = cornerCorrection(img)
    return img


# Show image
# plt.imshow(img)
# plt.show()
img = filter(num)
np.save('intermediate/Image.npy', img)
