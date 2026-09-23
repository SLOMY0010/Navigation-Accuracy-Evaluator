import cv2 as cv
import numpy as np

lower_yellow = np.array([0, 30, 60])
upper_yellow = np.array([40, 255, 255])

lower_white = np.array([0, 0, 130])
upper_white = np.array([179, 100, 255])

lower_gray = np.array([90, 0, 40])
upper_gray = np.array([170, 80, 100])

"""
Returns the mask that identifies the left yellow dashed line of the track.
Mask returned in BGR.
"""
def get_yellow_mask(img_bgr):
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    return mask

"""
Returns the mask that identifies the right white line of the track.
Mask returned in BGR.
"""
def get_white_mask(img_bgr):
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_white, upper_white)
    return mask