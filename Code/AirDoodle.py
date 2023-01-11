import numpy as np
import cv2 as cv
from collections import deque

def func(x):
    print(x)


def setMask(hsv, Lower_hsv, Upper_hsv, kernel):
    Mask = cv.inRange(hsv, Lower_hsv, Upper_hsv)
    Mask = cv.erode(Mask, kernel, iterations=1)
    Mask = cv.morphologyEx(Mask, cv.MORPH_OPEN, kernel)
    Mask = cv.dilate(Mask, kernel, iterations=1)
    return Mask


def preProcess(window):
    # window = np.float32(window)
    # window = cv.cvtColor(window, cv.COLOR_BGR2GRAY)
    # (thresh, window) = cv.threshold(window, 127, 255, cv.THRESH_BINARY)
    Img = np.asarray(window)
    # print(Img.shape)
    Img = Img[130:280, 250:400]
    return Img


def airDoodle():
    cv.namedWindow("TrackBars")
    cv.createTrackbar("Hue Max", "TrackBars", 138, 180, func)
    cv.createTrackbar("Sat Max", "TrackBars", 191, 255, func)
    cv.createTrackbar("Val Max", "TrackBars", 177, 255, func)
    cv.createTrackbar("Hue Min", "TrackBars", 113, 180, func)
    cv.createTrackbar("Sat Min", "TrackBars", 87, 255, func)
    cv.createTrackbar("Val Min", "TrackBars", 117, 255, func)

    index = 0
    kernel = np.ones((5, 5), np.uint8)

    Black_p = [deque(maxlen=1024)]
    color = (0, 0, 0)
    colorin = 0

    window = np.zeros((512, 800, 3))
    window.fill(255)
    # cv.rectangle(window, start_point, end_point, color, 2)
    # window = cv.rectangle(window, (20, 1), (110, 65), (0, 0, 0), 2)

    cv.putText(window, "CLEAR", (33, 37), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)
    # cv.namedWindow('Paint', cv.WINDOW_AUTOSIZE)
    start_point = (220, 100)
    end_point = (400, 250)

    cap = cv.VideoCapture(0)
    cap.set(3, 400)
    cap.set(4, 400)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv.flip(frame, 1)
            cv.putText(frame, "CLEAR", (33, 37), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)
            cv.rectangle(frame, start_point, end_point, color, 2)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            upper_hue = cv.getTrackbarPos("Hue Max", "TrackBars")
            upper_saturation = cv.getTrackbarPos("Sat Max", "TrackBars")
            upper_value = cv.getTrackbarPos("Val Max", "TrackBars")
            Lower_hue = cv.getTrackbarPos("Hue Min", "TrackBars")
            Lower_saturation = cv.getTrackbarPos("Sat Min", "TrackBars")
            Lower_value = cv.getTrackbarPos("Val Min", "TrackBars")
            Upper_hsv = np.array([upper_hue, upper_saturation, upper_value])
            Lower_hsv = np.array([Lower_hue, Lower_saturation, Lower_value])

            Mask = setMask(hsv, Lower_hsv, Upper_hsv, kernel)
            contours, _ = cv.findContours(Mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            center = None

            if len(contours) > 0:
                cnt = sorted(contours, key = cv.contourArea, reverse=True)[0]
                ((x, y), radius) = cv.minEnclosingCircle(cnt)
                cv.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                M = cv.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                if center[1] <= 65:
                    if 10 <= center[0] <= 120 and 1 <= center[1] <= 70:
                        Black_p = [deque(maxlen=512)]
                        index = 0
                        window[67:, :, :] = 0

                    elif 120 <= center[0] <= 630:
                        colorin = 0
                else:
                    if colorin == 0:
                        Black_p[index].appendleft(center)

            else:
                Black_p.append(deque(maxlen=512))
                index += 1

            points = [Black_p]
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv.line(frame, points[i][j][k - 1], points[i][j][k], color, 4, cv.LINE_AA)
                        cv.line(window, points[i][j][k - 1], points[i][j][k], color, 4)

            cv.imshow("Air Doodle", window)
            cv.imshow("Real time", frame)
            cv.imshow("mask", Mask)

            if cv.waitKey(1) == 27:
                break

    Img = preProcess(window)
    np.save('intermediate/Doodle.npy', Img)
    cv.imwrite('intermediate/Doodle.png', window)
    cap.release()
    cv.destroyAllWindows()

# airDoodle()