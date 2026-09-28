import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path
from sys import argv
from test import plot_images

CANNY_LOW = 20
CANNY_HIGH = 60
GAUSSIAN_KERNEL = (3, 3)
ROI_TOP = 80
Y_EVAL = 119


def main():
    if len(argv) < 2:
        print("Usage: python inspect_bottom.py <img_path>")
        return

    image_path = Path(argv[1])
    img = cv.imread(str(image_path))

    if img is None:
        print("Could not read image:", image_path)
        return

    roi_mask = get_bottom_roi_mask(img)
    bottom_edges = get_bottom_edges(img, roi_mask)
    left_points, right_points = get_boundary_points(bottom_edges)

    roi_view = cv.bitwise_and(img, img, mask=roi_mask)

    debug_img = img.copy()

    # Start of bottom ROI
    cv.line(debug_img, (0, ROI_TOP), (img.shape[1] - 1, ROI_TOP), (0, 255, 255), 1)

    # Bottom evaluation row
    cv.line(debug_img, (0, Y_EVAL), (img.shape[1] - 1, Y_EVAL), (0, 0, 255), 1)

    points_img = img.copy()
    for x, y in left_points:
        cv.circle(points_img, (x, y), 2, (0, 255, 0), -1)

    for x, y in right_points:
        cv.circle(points_img, (x, y), 2, (255, 0, 0), -1)

    plot_images(img, roi_view, bottom_edges, points_img, titles=[
        "Original",
        "Bottom ROI",
        "Bottom Canny",
        "Boundary points"
    ])


def get_boundary_points(edges):
    left_points = []
    right_points = []

    camera_center = (edges.shape[1] - 1) / 2

    for y in range(ROI_TOP, Y_EVAL + 1):
        xs = np.where(edges[y] > 0)[0]

        left_xs = xs[xs < camera_center]
        right_xs = xs[xs > camera_center]

        if len(left_xs) > 0:
            x_left = left_xs.max()
            left_points.append((x_left, y))

        if len(right_xs) > 0:
            x_right = right_xs.min()
            right_points.append((x_right, y))

    return left_points, right_points


def get_bottom_edges(img, roi_mask):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    edges = cv.Canny(blur, CANNY_LOW, CANNY_HIGH)

    return cv.bitwise_and(edges, roi_mask)


def get_bottom_roi_mask(img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[ROI_TOP:Y_EVAL + 1, :] = 255
    return mask


if __name__ == "__main__":
    main()