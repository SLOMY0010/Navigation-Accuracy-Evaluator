import numpy as np

# Those thresholds are used to detect the yellow dashed lane of the track
YELLOW_MIN_HSV = np.array([0, 30, 60])
YELLOW_MAX_HSV = np.array([40, 255, 255])
# Hue wrap caused by the pink tint of the camera
YELLOW_WRAP_MIN_HSV = np.array([175, 30, 60])
YELLOW_WRAP_MAX_HSV = np.array([179, 255, 255])

# Canny edge algorithm thresholds
CANNY_LOW = 40
CANNY_HIGH = 110
GAUSSIAN_KERNEL = (5, 5)

MIN_LANE_WIDTH = 72 # Values taken from many real sample images, change only based on samples
MAX_LANE_WIDTH = 88 

# Reference row to measure the lane width from and evaluate
Y_REF = 80
MIN_LEFT_Y_SPAN = 20
MIN_LEFT_INLIERS = 5
