import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
from sys import argv
from pathlib import Path
import csv
from config import CANNY_HIGH, CANNY_LOW, GUASSIAN_KERNEL, MAX_LANE_WIDTH, MIN_LANE_WIDTH, MIN_LEFT_INLIERS, MIN_LEFT_Y_SPAN, Y_REF, YELLOW_MAX_HSV, YELLOW_MIN_HSV, YELLOW_WRAP_MAX_HSV, YELLOW_WRAP_MIN_HSV


def main():
    condition_dir = Path(argv[1])

    output_dir = Path("/mnt/e/Thesis_experimentation/nav-acc/results") / condition_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_dir = output_dir / "failed_frames"

    results = process_lighting_condition(condition_dir, failure_dir)

    save_frame_results_csv(results, output_dir / "frame_results.csv")

    tub_summaries = get_tub_summaries(results)

    save_tub_summaries_csv(tub_summaries, output_dir / "tub_summary.csv")

    summarize_results(results)

    print(f"\nResults saved to: {output_dir}")


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


def save_failure_debug(img, img_path, failure_reason, failure_root, tub_name,left_curve=None, left_edge_points=None, inliers=None, right_candidates=None, x_left=None):

    if left_edge_points is None:
        left_edge_points = []

    debug_img = draw_left_curve(img, left_curve, left_edge_points, inliers)

    # Draw reference row
    cv.line(debug_img, (0, Y_REF), (img.shape[1] - 1, Y_REF), (0, 255, 255), 1)

    # Draw Canny candidates at reference row
    if right_candidates is not None:
        edge_xs = np.where(right_candidates[Y_REF] > 0)[0]

        for x in edge_xs:
            cv.circle(debug_img, (int(x), Y_REF), 2, (0, 0, 255), -1)

        # Draw grouped Canny edges slightly larger
        grouped_xs = group_xs(edge_xs)

        for x in grouped_xs:
            cv.circle(debug_img, (int(round(x)), Y_REF), 3, (255, 0, 255), 1)

    # Draw left boudnary at reference row
    if x_left is not None:
        x_left_int = int(round(x_left))

        if 0 <= x_left_int < img.shape[1]:
            cv.circle(debug_img, (x_left_int, Y_REF), 4, (255, 255, 0), -1)

    # Put failure reason on the image
    text = f"FAIL: {failure_reason}"    
    cv.putText(debug_img, text, (5, 15), cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
    cv.putText(debug_img, text, (4, 14), cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    tub_dir = Path(failure_root) / tub_name
    tub_dir.mkdir(parents=True, exist_ok=True)

    output_path = tub_dir / (f"{img_path.stem}___{failure_reason}.jpg")
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
    roi_mask = np.zeros_like(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    polygon = np.array([
        [0, 119],
        [0, 65],
        [20, 45],
        [140, 45],
        [159, 65],
        [159, 119]
    ], dtype=np.int32)
    cv.fillPoly(roi_mask, [polygon], 255)

    left_candidates = get_left_candidates(img, roi_mask)
    right_candidates = get_right_candidates(img, roi_mask)

    # ---------- LEFT LANE DETECTION ---------- #

    
    filtered_left_mask, labels, stats, accepted_labels = filter_left_components(left_candidates)
    
    left_edge_points = get_left_edge_points(labels, stats, accepted_labels)

    left_curve, inliers = get_left_boundary(left_edge_points)

    left_failure_reason = validate_left_boundary(left_edge_points, inliers, left_curve, img.shape[1])
    if left_failure_reason is not None:
        save_failure_debug(
            img,
            img_path,
            left_failure_reason,
            failure_root,
            tub_name,
            left_curve=left_curve,
            left_edge_points=left_edge_points,
            inliers=inliers,
            right_candidates=right_candidates
        )
        return None, left_failure_reason

    x_left = np.polyval(left_curve, Y_REF)



    # ---------- RIGHT LANE DETECTION ---------- #
    

    edge_xs = np.where(right_candidates[Y_REF] > 0)[0]
    grouped_xs = group_xs(edge_xs)
    lane_widths = grouped_xs - x_left
    valid = ((lane_widths >= MIN_LANE_WIDTH) & (lane_widths <= MAX_LANE_WIDTH))
    valid_right_edges = grouped_xs[valid]

    if len(valid_right_edges)  == 0:
        save_failure_debug(
            img,
            img_path,
            "right_no_valid_canny_edges",
            failure_root,
            tub_name,
            left_curve=left_curve,
            left_edge_points=left_edge_points,
            inliers=inliers,
            right_candidates=right_candidates,
            x_left=x_left
        )

        return None, "right_no_valid_canny_edges"

    x_right = valid_right_edges[0]



    # ---------- EVALUATION ---------- #
    
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


def group_xs(xs):
    """
    Groups adjacent pixels and returns the mean of each group in an array.
    e.g.:
    [100, 101, 120, 121, 130] -> [[100, 101], [120, 121], [130]] -> [100.5, 120.5, 130]
    """

    if len(xs) == 0:
        return np.array([])
    
    groups = []
    current_group = [xs[0]]

    for x in xs[1:]:

        # If the point is adjacent to the previous one, add it to the group
        if x - current_group[-1] == 1:
            current_group.append(x)
        else:
            groups.append(current_group)
            current_group = [x]

    groups.append(current_group)
    grouped_xs = np.array([np.mean(group) for group in groups])
    return grouped_xs


def draw_left_curve(img, curve, edge_points, inliers):
    ransac_img = img.copy()
    if inliers is not None:
        for i, (x, y) in enumerate(edge_points):

            if inliers[i]:
                # Green = trusted
                color = (0, 255, 0)
            else:
                # Red = rejected
                color = (0, 0, 255)

            cv.circle(ransac_img, (x, y), 1, color, -1)

    if curve is not None:
        y_values = np.arange(45, 120) # Starts from 45 because this is the beginning of our ROI mask
        x_values = np.polyval(curve, y_values)

        # Only draw positions that are actually inside the image
        valid = ((x_values >= 0) & (x_values < img.shape[1]))

        curve_points = np.column_stack((x_values[valid], y_values[valid])).astype(np.int32)

        # Connect curve_points
        cv.polylines(ransac_img, [curve_points.reshape(-1, 1, 2)], False, (255, 0, 0), 1)

    return ransac_img



def validate_left_boundary(edge_points, inliers, left_curve, image_width):
    if inliers is None or left_curve is None:
        return "left_no_ransac_model"

    points = np.array(edge_points, dtype=np.float64)
    inlier_points = points[inliers]

    if len(inlier_points) < MIN_LEFT_INLIERS:
        return "left_too_few_inliers"

    inlier_ys = inlier_points[:, 1]
    y_min = np.min(inlier_ys)
    y_max = np.max(inlier_ys)
    y_span = y_max - y_min
    if y_span < MIN_LEFT_Y_SPAN:
        return "left_insufficient_y_span"

    # If Y_REF is inside the observed range, this is interpolation which is totally fine
    if y_min <= Y_REF <= y_max:
        return None

    if Y_REF < y_min:
        extrapolation = y_min - Y_REF
    else:
        extrapolation = Y_REF - y_max

    if extrapolation > y_span:
        return "left_excessive_extrapolation"

    return None


def get_left_boundary(edge_points, iterations=200, residual_threshold=3.0):

    # 3 points are required to fit a curve
    if len(edge_points) < 3:
        return None, None

    points = np.array(edge_points, dtype=np.float64)

    xs = points[:, 0]
    ys = points[:, 1]

    rng = np.random.default_rng(42)

    best_inliers = None
    best_count = 0
    best_mean_residual = np.inf

    for _ in range(iterations):

        # Pick 3 sample points
        sample_points = rng.choice(len(points), size=3, replace=False)
        sample_xs = xs[sample_points]
        sample_ys = ys[sample_points]

        # The points must not be in the same y level
        if len(np.unique(sample_ys)) < 3:
            continue

        # Fit a curve through the sample points
        coefficients = np.polyfit(sample_ys, sample_xs, 2)

        # Predict xs for all actual ys
        predicted_xs = np.polyval(coefficients, ys)

        # Calculate residual
        residuals = np.abs(predicted_xs - xs)

        inliers = residuals <= residual_threshold

        count = np.sum(inliers)

        if count < 3:
            continue

        mean_residual = np.mean(residuals[inliers])

        # Criteria: largest inliers, if tie, smallest error
        if count > best_count or count == best_count and mean_residual < best_mean_residual:
            best_count = count
            best_inliers = inliers
            best_mean_residual = mean_residual

    if best_inliers is None:
        return None, None

    # Trusted RANSAC points are found, now fit through all of them:
    final_coefficients = np.polyfit(ys[best_inliers], xs[best_inliers], 2)

    return final_coefficients, best_inliers


def get_left_edge_points(labels, stats, accepted_labels):
    edge_points = []

    for label_id in accepted_labels:

        y_start = stats[label_id, cv.CC_STAT_TOP]
        height = stats[label_id, cv.CC_STAT_HEIGHT]

        for y in range(y_start, y_start + height):

            xs = np.where(labels[y] == label_id)[0]

            if len(xs) > 0:
                x = xs.max()   # rightmost pixel = road-facing edge
                edge_points.append((x, y))

    return edge_points


def filter_left_components(left_mask, min_area=3):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(left_mask, connectivity=8)

    filtered_mask = np.zeros_like(left_mask)
    accepted_labels = []

    for i in range(1, num_labels):

        area = stats[i, cv.CC_STAT_AREA]
        if area < min_area:
            continue

        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        aspect_ratio = w / h

        cx, cy = centroids[i]

        # Reject components in the far right side
        if cx > left_mask.shape[1] * 0.65:
            continue

        if aspect_ratio <= 1.5:
            filtered_mask[labels == i] = 255
            accepted_labels.append(i)

    return filtered_mask, labels, stats, accepted_labels

def draw_components(img_bgr, components):

    output = img_bgr.copy()

    for component in components:
        x, y, w, h = component["bbox"]
        cx, cy = component["centroid"]

        cv.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 1)
        cv.circle(output, (int(cx), int(cy)), 3, (0, 255, 0), -1)

    return output


def get_left_candidates(img_bgr, roi_mask):
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    yellow_mask = cv.inRange(hsv, YELLOW_MIN_HSV, YELLOW_MAX_HSV)
    wrap_mask = cv.inRange(hsv, YELLOW_WRAP_MIN_HSV, YELLOW_WRAP_MAX_HSV)
    mask = cv.bitwise_or(yellow_mask, wrap_mask)
    return cv.bitwise_and(mask, roi_mask)

def get_right_candidates(img_bgr, roi_mask):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, GUASSIAN_KERNEL, 0)
    canny_edges = cv.Canny(blur, threshold1=CANNY_LOW, threshold2=CANNY_HIGH)

    return cv.bitwise_and(canny_edges, roi_mask)


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


def darken(img, alpha=1.00):
    return cv.convertScaleAbs(img, alpha=alpha)


if __name__ == '__main__':
    main()