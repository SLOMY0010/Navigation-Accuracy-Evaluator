import cv2 as cv
import numpy as np
from pathlib import Path
from sys import argv

from test import (
    get_left_candidates,
    get_right_candidates,
    filter_left_components,
    get_left_edge_points,
    get_left_boundary,
    validate_left_boundary,
    group_xs,
    calculate_lane_error,
    draw_left_curve,
    plot_images,
    Y_REF,
    MIN_LANE_WIDTH,
    MAX_LANE_WIDTH
)


def main():
    if len(argv) < 2:
        print("Usage: python inspect_image.py <image_path>")
        return

    image_path = Path(argv[1])

    img = cv.imread(str(image_path))

    if img is None:
        print("Could not read image:", image_path)
        return

    # ---------------- ROI ---------------- #

    roi_mask = np.zeros_like(
        cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    )

    polygon = np.array([
        [0, 119],
        [0, 65],
        [20, 45],
        [140, 45],
        [159, 65],
        [159, 119]
    ], dtype=np.int32)

    cv.fillPoly(roi_mask, [polygon], 255)


    # ---------------- LEFT ---------------- #

    left_candidates = get_left_candidates(
        img,
        roi_mask
    )

    filtered_left_mask, labels, stats, accepted_labels = \
        filter_left_components(left_candidates)

    left_edge_points = get_left_edge_points(
        labels,
        stats,
        accepted_labels
    )

    left_curve, inliers = get_left_boundary(
        left_edge_points
    )

    left_failure = validate_left_boundary(
        left_edge_points,
        inliers,
        left_curve,
        img.shape[1]
    )


    # ---------------- RIGHT ---------------- #

    right_candidates = get_right_candidates(
        img,
        roi_mask
    )


    # ---------------- DEBUG IMAGE ---------------- #

    debug_img = draw_left_curve(
        img,
        left_curve,
        left_edge_points,
        inliers
    )

    # Reference row
    cv.line(
        debug_img,
        (0, Y_REF),
        (img.shape[1] - 1, Y_REF),
        (0, 255, 255),
        1
    )


    # ---------------- CHECK LEFT ---------------- #

    if left_failure is not None:
        print("LEFT FAILED:", left_failure)

        plot_images(
            img,
            left_candidates,
            filtered_left_mask,
            right_candidates,
            debug_img,
            titles=[
                "Original",
                "Raw yellow mask",
                "Filtered yellow mask",
                "Canny",
                f"FAIL: {left_failure}"
            ]
        )

        return


    x_left = np.polyval(
        left_curve,
        Y_REF
    )

    # Draw x_left
    cv.circle(
        debug_img,
        (int(round(x_left)), Y_REF),
        4,
        (255, 255, 0),
        -1
    )


    # ---------------- FIND RIGHT ---------------- #

    edge_xs = np.where(
        right_candidates[Y_REF] > 0
    )[0]

    grouped_xs = group_xs(edge_xs)

    # Draw every grouped Canny edge
    for x in grouped_xs:
        cv.circle(
            debug_img,
            (int(round(x)), Y_REF),
            3,
            (0, 0, 255),
            -1
        )

    lane_widths = grouped_xs - x_left

    valid = (
        (lane_widths >= MIN_LANE_WIDTH) &
        (lane_widths <= MAX_LANE_WIDTH)
    )

    valid_right_edges = grouped_xs[valid]

    if len(valid_right_edges) == 0:
        print("RIGHT FAILED: right_no_valid_canny_edges")

        print("x_left:", x_left)
        print("Canny groups:", grouped_xs)
        print("Possible widths:", lane_widths)

        plot_images(
            img,
            left_candidates,
            filtered_left_mask,
            right_candidates,
            debug_img,
            titles=[
                "Original",
                "Raw yellow mask",
                "Filtered yellow mask",
                "Canny",
                "Right detection failed"
            ]
        )

        return


    # ---------------- FINAL RESULT ---------------- #

    x_right = valid_right_edges[0]

    cv.circle(
        debug_img,
        (int(round(x_right)), Y_REF),
        5,
        (0, 255, 0),
        -1
    )

    result = calculate_lane_error(
        x_left,
        x_right,
        img.shape[1]
    )

    lane_center = result["lane_center"]
    camera_center = result["camera_center"]

    # Lane center
    cv.circle(
        debug_img,
        (int(round(lane_center)), Y_REF),
        4,
        (255, 0, 255),
        -1
    )

    # Camera center
    cv.circle(
        debug_img,
        (int(round(camera_center)), Y_REF),
        4,
        (255, 255, 255),
        -1
    )

    print()
    print("----- IMAGE RESULT -----")
    print("Image:", image_path.name)
    print(f"x_left: {x_left:.2f}")
    print(f"x_right: {x_right:.2f}")
    print(f"Lane width: {result['lane_width']:.2f}")
    print(f"Lane center: {lane_center:.2f}")
    print(f"Camera center: {camera_center:.2f}")
    print(
        f"Signed normalized error: "
        f"{result['signed_normalized_error']:.3f}"
    )
    print(
        f"Absolute normalized error: "
        f"{result['absolute_normalized_error']:.3f}"
    )
    print(
        f"Squared pixel error: "
        f"{result['squared_pixel_error']:.3f}"
    )

    plot_images(
        img,
        cv.bitwise_and(cv.cvtColor(img, cv.COLOR_BGR2GRAY), roi_mask),
        left_candidates,
        filtered_left_mask,
        right_candidates,
        debug_img,
        titles=[
            "Original",
            "ROI mask",
            "Raw yellow mask",
            "Filtered yellow mask",
            "Canny",
            "Final detection"
        ]
    )


if __name__ == "__main__":
    main()