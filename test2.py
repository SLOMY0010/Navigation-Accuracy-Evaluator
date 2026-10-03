import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
from sys import argv, exit
from pathlib import Path
import csv
from config import CANNY_HIGH, CANNY_LOW, GAUSSIAN_KERNEL

ROI_TOP = 75
Y_EVAL = 119
MIN_LANE_WIDTH = 118
MAX_LANE_WIDTH = 145
CENTER_MARGIN = 5 # px

# fail_limit = 7

def main():
    condition_dir = Path(argv[1])

    output_dir = Path("/mnt/e/Thesis_experimentation/nav-acc/results_bottom") / condition_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_dir = output_dir / "failed_frames"

    results = process_lighting_condition(condition_dir, failure_dir)

    save_frame_results_csv(results, output_dir / "frame_results.csv")

    tub_summaries = get_tub_summaries(results)

    save_tub_summaries_csv(tub_summaries, output_dir / "tub_summary.csv")

    summarize_results(results)

    print(f"\nResults saved to: {output_dir}")


def fit_boundary_line(points, iterations=200, residual_threshold=2.0):
    if len(points) < 2:
        return None, None

    points = np.array(points, dtype=np.float64)

    xs = points[:, 0]
    ys = points[:, 1]

    rng = np.random.default_rng(42)

    best_inliers = None
    best_count = 0
    best_mean_residual = np.inf

    for _ in range(iterations):

        # Pick two random points
        sample_indicies = rng.choice(len(points), size=2, replace=False)
        sample_xs = xs[sample_indicies]        
        sample_ys = ys[sample_indicies]

        if sample_ys[0] == sample_ys[1]:
            continue

        # Fit a line
        coefficients = np.polyfit(sample_ys, sample_xs, 1)

        # Predict x for every observed y
        predicted_xs = np.polyval(coefficients, ys)

        residuals = np.abs(predicted_xs - xs)

        inliers = residuals <= residual_threshold

        count = np.sum(inliers)

        if count < 2:
            continue

        mean_residual = np.mean(residuals[inliers])


        if count > best_count or (count == best_count and mean_residual < best_mean_residual):
            best_count = count
            best_inliers = inliers
            best_mean_residual = mean_residual

    if best_inliers is None:
        return None, None

    # RANSAC found the trusted points, refit using all of them
    final_coefficients = np.polyfit(ys[best_inliers], xs[best_inliers], 1)            

    return final_coefficients, best_inliers
   

def get_boundary_points(edges):
    left_points = []
    right_points = []

    camera_center = (edges.shape[1] - 1) / 2

    for y in range(ROI_TOP, Y_EVAL + 1):
        xs = np.where(edges[y] > 0)[0]

        left_xs = xs[xs < camera_center - CENTER_MARGIN]
        right_xs = xs[xs > camera_center + CENTER_MARGIN]

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



def save_frame_results_csv(results, filepath):
    fieldnames = [
        "tub",
        "image",
        "valid",
        "failure_reason",
        "x_left",
        "x_right",
        "lane_width",
        "lane_center",
        "camera_center",
        "signed_pixel_error",
        "absolute_pixel_error",
        "squared_pixel_error",
        "signed_normalized_error",
        "absolute_normalized_error"
    ]

    with open(filepath, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_tub_summaries_csv(summaries, filepath):
    fieldnames = [
        "tub",
        "total_frames",
        "valid_detections",
        "detection_failures",
        "failure_rate",
        "mean_absolute_normalized_error",
        "median_absolute_normalized_error",
        "mean_squared_pixel_error"
    ]

    with open(filepath, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def save_failure_debug(img, img_path, failure_reason, failure_root, tub_name,left_points=None, right_points=None, left_line=None, right_line=None,left_inliers=None, right_inliers=None):
    debug_img = img.copy()

    if left_points is None:
        left_points = []
    if right_points is None:
        right_points = []

    # Draw RANSAC points
    if left_inliers is not None:
        for i, (x, y) in enumerate(left_points):
            if left_inliers[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv.circle(debug_img, (x, y), 1, color, -1)
    if right_inliers is not None:
        for i, (x, y) in enumerate(right_points):
            if right_inliers[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv.circle(debug_img, (x, y), 1, color, -1)

    # Draw fitted lines
    y_values = np.arange(ROI_TOP, Y_EVAL + 1)
    if left_line is not None:
        x_values = np.polyval(left_line, y_values)
        valid = (x_values >= 0) & (x_values < img.shape[1])

        line_points = np.column_stack((x_values[valid], y_values[valid])).astype(np.int32)

        if len(line_points) >= 2:
            cv.polylines(debug_img, [line_points.reshape(-1, 1, 2)], False, (255, 0, 0), 1)

    if right_line is not None:
        x_values = np.polyval(right_line, y_values)
        valid = (x_values >= 0) & (x_values < img.shape[1])

        line_points = np.column_stack((x_values[valid], y_values[valid])).astype(np.int32)

        if len(line_points) >= 2:
            cv.polylines(debug_img, [line_points.reshape(-1, 1, 2)], False, (255, 0, 0), 1)

    # Failure text
    text = f"FAIL: {failure_reason}"

    cv.putText(
        debug_img, text, (5, 15), cv.FONT_HERSHEY_PLAIN, 0.30, (0, 0, 0), 2, lineType=cv.LINE_AA)

    cv.putText(
        debug_img, text, (4, 14), cv.FONT_HERSHEY_PLAIN, 0.30, (255, 255, 255), 1, lineType=cv.LINE_AA)

    tub_dir = Path(failure_root) / tub_name

    tub_dir.mkdir(parents=True, exist_ok=True)

    output_path = (tub_dir / f"{img_path.stem}___{failure_reason}.jpg")
    cv.imwrite(str(output_path), debug_img)


def get_tub_summaries(results):
    tubs = {}

    for result in results:
        tub = result["tub"]

        if tub not in tubs:
            tubs[tub] = []

        tubs[tub].append(result)

    summaries = []

    for tub, tub_results in sorted(tubs.items()):
        summary = calculate_summary(tub_results)
        summary["tub"] = tub
        summaries.append(summary)

    return summaries


def calculate_summary(results):
    valid_results = [r for r in results if r["valid"]]

    total = len(results)
    valid = len(valid_results)
    failed = total - valid

    if valid > 0:
        errors = np.array([r['absolute_normalized_error'] for r in valid_results])
        squared_pixel_errors = np.array([r["squared_pixel_error"] for r in valid_results])

        mean_error = np.mean(errors)
        median_error = np.median(errors)
        mean_squared_error = np.mean(squared_pixel_errors)
        
    else:
        mean_error = None
        median_error = None
        mean_squared_error = None

    return {
        "total_frames": total,
        "valid_detections": valid,
        "detection_failures": failed,
        "failure_rate": failed / total * 100 if total > 0 else 0,
        "mean_absolute_normalized_error": mean_error,
        "median_absolute_normalized_error": median_error,
        "mean_squared_pixel_error": mean_squared_error
    }


def summarize_results(results):
    valid_results = [r for r in results if r["valid"]]

    total = len(results)
    valid = len(valid_results)
    failed = total - valid

    if valid == 0:
        print("No valid frames.")
        return

    errors = np.array([r["absolute_normalized_error"] for r in valid_results])
    squared_pixel_errors = np.array([r["squared_pixel_error"] for r in valid_results])
    print()
    print("----- CONDITION SUMMARY -----")
    print(f"Total frames: {total}")
    print(f"Valid detections: {valid}")
    print(f"Detection failures: {failed}")
    print(
        f"Failure rate: "
        f"{failed / total * 100:.2f}%"
    )

    print(
        f"Mean absolute normalized error: "
        f"{np.mean(errors):.3f}"
    )

    print(
        f"Median absolute normalized error: "
        f"{np.median(errors):.3f}"
    )

    print(
        f"Mean squared pixel error: "
        f"{np.mean(squared_pixel_errors):.3f}"
    )


def process_lighting_condition(directory, failure_root):
    condition_dir = Path(directory)

    image_paths = sorted(condition_dir.rglob("*_cam_image_array_.jpg"))

    if len(image_paths) == 0:
        print("No DonkeyCar images found.")
        return []

    results = []

    for i, image_path in enumerate(image_paths, start=1):

        tub_name = image_path.relative_to(condition_dir).parts[0]
        result, failure_reason = evaluate_frame(image_path, failure_root=failure_root, tub_name=tub_name)

        row = {
            "tub": tub_name,
            "image": image_path.name,
            "valid": result is not None,
            "failure_reason": failure_reason
        }

        # Insert the result dictionary
        if result is not None:
            row.update(result)

        results.append(row)

        if i % 250 == 0:
            print(f"Processed {i}/{len(image_paths)} frames...")

    return results


def evaluate_frame(img_path, failure_root=None, tub_name=None):
    """
    Evaluates visual normalized lane deviation.
    Returns to variables: results, failure_reason; in case of successful evaluation, the latter is None.
    """
    img = cv.imread(str(img_path))
    if img is None:
        return None, "image_read_failed"

    # Region of Interest (ROI) mask
    roi_mask = get_bottom_roi_mask(img)

    bottom_edges = get_bottom_edges(img, roi_mask)

    # ---------- BOUNDARY POINTS ---------- #

    left_points, right_points = get_boundary_points(bottom_edges)

    # ---------- FIT STRAIGHT LINES ---------- #
    
    left_line, left_inliers = fit_boundary_line(left_points)
    right_line, right_inliers = fit_boundary_line(right_points)

    if left_line is None:
        save_failure_debug(
            img,
            img_path,
            "left_line_failed",
            failure_root,
            tub_name,
            left_points=left_points,
            right_points=right_points,
            left_line=left_line,
            right_line=right_line,
            left_inliers=left_inliers,
            right_inliers=right_inliers
        )

        return None, "left_line_failed"
    if right_line is None:
        save_failure_debug(
            img,
            img_path,
            "right_line_failed",
            failure_root,
            tub_name,
            left_points=left_points,
            right_points=right_points,
            left_line=left_line,
            right_line=right_line,
            left_inliers=left_inliers,
            right_inliers=right_inliers
        )
        return None, "right_line_failed"

    # ---------- BOTTOM INTERSECTIONS ---------- #
    
    x_left = np.polyval(left_line, Y_EVAL)
    x_right = np.polyval(right_line, Y_EVAL)

    if x_right <= x_left:
        save_failure_debug(
            img,
            img_path,
            "invalid_lane_geometry",
            failure_root,
            tub_name,
            left_points=left_points,
            right_points=right_points,
            left_line=left_line,
            right_line=right_line,
            left_inliers=left_inliers,
            right_inliers=right_inliers
        )
        return None, "invalid_lane_geometry"

    lane_width = x_right - x_left
    if not (MIN_LANE_WIDTH <= lane_width <= MAX_LANE_WIDTH):
        # print(
        #     img_path.name,
        #     f"x_left={x_left:.2f}",
        #     f"x_right={x_right:.2f}",
        #     f"width={lane_width:.2f}",
        #     f"Limits: {MIN_LANE_WIDTH} - {MAX_LANE_WIDTH}"
        # )
        # if fail_limit == 0:
        #     exit("gg bro")
        # fail_limit -= 1
        save_failure_debug(
            img,
            img_path,
            f"lane_width_out_of_range_({lane_width})",
            failure_root,
            tub_name,
            left_points=left_points,
            right_points=right_points,
            left_line=left_line,
            right_line=right_line,
            left_inliers=left_inliers,
            right_inliers=right_inliers
        )
        return None, "lane_width_out_of_range"

    # ---------- EVALUATION ----------- #

    result = calculate_lane_error(x_left, x_right, img.shape[1])

    return result, None


def calculate_lane_error(x_left, x_right, image_width):
    """
    eN = (xCamera - xLane) / (WLane / 2)
    """
    lane_width = x_right - x_left

    lane_center = (x_left + x_right) / 2
    camera_center = (image_width - 1) / 2

    signed_pixel_error = camera_center - lane_center
    absolute_pixel_error = abs(signed_pixel_error)

    squared_pixel_error = signed_pixel_error ** 2

    half_lane_width = lane_width / 2

    signed_normalized_error = signed_pixel_error / half_lane_width
    absolute_normalized_error = abs(signed_normalized_error)

    return {
        "x_left": x_left,
        "x_right": x_right,
        "lane_width": lane_width,
        "lane_center": lane_center,
        "camera_center": camera_center,
        "signed_pixel_error": signed_pixel_error,
        "absolute_pixel_error": absolute_pixel_error,
        "squared_pixel_error": squared_pixel_error,
        "signed_normalized_error": signed_normalized_error,
        "absolute_normalized_error": absolute_normalized_error
    }


def plot_images(*images, titles=None, cols=3):
    n = len(images)

    if n == 0:
        return

    cols = min(cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4 * rows),
        squeeze=False
    )

    axes = axes.flatten()

    for i, img in enumerate(images):

        # Grayscale image: masks, Canny output, grayscale, etc.
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap="gray", vmin=0, vmax=255)

        # Normal OpenCV color image (BGR)
        else:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)

        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])

        axes[i].axis("off")

    # Hide unused subplot spaces
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()