import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

from pathlib import Path
from sys import argv
from test import plot_images
from test2 import get_bottom_edges, get_bottom_roi_mask, get_boundary_points, CANNY_HIGH, CANNY_LOW, Y_EVAL, ROI_TOP, fit_boundary_line


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

    left_line, left_inliers = fit_boundary_line(left_points)
    right_line, right_inliers = fit_boundary_line(right_points)

    roi_view = cv.bitwise_and(img, img, mask=roi_mask)

    debug_img = img.copy()

    if left_inliers is not None:
        for i, (x, y) in enumerate(left_points):
            if left_inliers[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv.circle(debug_img, (x, y), 2, color, -1)

    if right_inliers is not None:
        for i, (x, y) in enumerate(right_points):
            if right_inliers[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv.circle(debug_img, (x, y), 2, color, -1)

    y_values = np.arange(ROI_TOP, Y_EVAL + 1)

    if left_line is not None:
        x_values = np.polyval(left_line, y_values)
        valid = (x_values >= 0) & (x_values < img.shape[1])
        line_points = np.column_stack([x_values[valid], y_values[valid]]).astype(np.int32)

        cv.polylines(debug_img, [line_points.reshape(-1, 1, 2)], False, (255, 0, 0), 1)        

    if right_line is not None:
        x_values = np.polyval(right_line, y_values)
        valid = (x_values >= 0) & (x_values < img.shape[1])
        line_points = np.column_stack([x_values[valid], y_values[valid]]).astype(np.int32)

        cv.polylines(debug_img, [line_points.reshape(-1, 1, 2)], False, (255, 0, 0), 1)        

    if left_line is not None:
        x_left_bottom = np.polyval(left_line, Y_EVAL)
        print(f"xL: {x_left_bottom:.2f}")

    if right_line is not None:
        x_right_bottom = np.polyval(right_line, Y_EVAL)
        print(f"xR: {x_right_bottom:.2f}")

    # Start of bottom ROI
    cv.line(debug_img, (0, ROI_TOP), (img.shape[1] - 1, ROI_TOP), (0, 255, 255), 1)

    # Bottom evaluation row
    cv.line(debug_img, (0, Y_EVAL), (img.shape[1] - 1, Y_EVAL), (0, 255, 255), 1)

    points_img = img.copy()
    for x, y in left_points:
        cv.circle(points_img, (x, y), 2, (0, 255, 0), -1)

    for x, y in right_points:
        cv.circle(points_img, (x, y), 2, (0, 255, 255), -1)

    plot_images(img, roi_view, bottom_edges, points_img, debug_img,titles=[
        "Original",
        "Bottom ROI",
        "Bottom Canny",
        "Boundary points",
        "Fitted lines"
    ])


if __name__ == "__main__":
    main()