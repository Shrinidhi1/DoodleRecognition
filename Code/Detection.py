import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

def nothing(x):
    pass


# Orders the point of rectangle in ABCD order
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def Preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 1000)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    return thresh


def FindContours(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def SelectContours(contours):
    perimeters = [cv2.arcLength(contours[i], True) for i in range(len(contours))]
    listindex = [i for i in range(len(perimeters)) if (perimeters[i] > perimeters[0] / 7)] # and perimeters[i] < perimeters[0] / 1.5)]
    return perimeters, listindex

def detection():
    cv2.namedWindow('Tracker')
    cv2.createTrackbar('min', 'Tracker', 100, 255, nothing)
    cv2.createTrackbar('max', 'Tracker', 200, 255, nothing)
    cv2.createTrackbar('Precision', 'Tracker', 80, 255, nothing)

    while True:
        _, frame = cap.read()

        thresh = Preprocess(frame)
        contours = FindContours(thresh)
        perimeters, listindex = SelectContours(contours)

    # Draw Contour and Perspective correction
        warp = False
        imgcont = frame.copy()
        if len(listindex) != 0:
            cv2.drawContours(imgcont, [contours[listindex[0]]], 0, (0, 255, 0), 5)

            peri = cv2.arcLength(contours[listindex[0]], True)
            approx = cv2.approxPolyDP(contours[listindex[0]], 0.015 * peri, True)

            x = np.zeros((4, 2))
            if len(approx) == 4:
                screenCnt = approx
                pts = screenCnt.reshape(4, 2)
                if np.sum(pts) != 0:
                    x = order_points(pts)

            warped = four_point_transform(frame, x)
            warp = True

        cv2.imshow('Image', imgcont)
        if warp:
            nw = warped.copy()
            cv2.imshow('Warped', warped)

        key = cv2.waitKey(1)
        if key == 27:  # Escape key
        # finalImg = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
           finalImg = warped.copy()
           break

    Img = np.asarray(finalImg)

    dim = (28, 28)
    Img = cv2.resize(Img, dim, interpolation=cv2.INTER_AREA)

    cap.release()
    cv2.destroyAllWindows()

    # plt.imshow(Img, cmap='gray')
    np.save('intermediate/Doodle.npy', Img)
    # plt.savefig('intermediate/Doodle.png')
    Img = np.array(Img)
    # print(Img.shape)
    height, width = Img.shape[:2]
    Img = cv2.resize(Img, (7 * width, 7 * height), interpolation=cv2.INTER_CUBIC)
    # Img = np.resize(Img, (64, 64, 3))
    cv2.imwrite('intermediate/Doodle.png', Img)
    # plt.show()

# detection()
