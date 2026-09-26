import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

YELLOW_MIN_HSV = np.array([0, 30, 60])
YELLOW_MAX_HSV = np.array([40, 255, 255])
# Hue wrap caused by the pink tint of the camera
YELLOW_WRAP_MIN_HSV = np.array([175, 30, 60])
YELLOW_WRAP_MAX_HSV = np.array([179, 255, 255])

CANNY_LOW = 20
CANNY_HIGH = 60
GUASSIAN_KERNEL = (3, 3)

Y_REF = 80
MIN_LEFT_Y_SPAN = 20


def main():
    img = cv.imread("imgs/curves/6.jpg")

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

    alpha = 1.0
    img = darken(img, alpha=alpha)

    left_candidates = get_left_candidates(img, roi_mask)
    right_candidates = get_right_candidates(img, roi_mask)

    filtered_left_mask, labels, stats, accepted_labels = filter_left_components(left_candidates)
    left_edge_points = get_left_edge_points(labels, stats, accepted_labels)

    left_curve, inliers = get_left_boundary(left_edge_points)

    left_valid = validate_left_boundary(left_edge_points, inliers)
    if left_valid:
        x_left = np.polyval(left_curve, Y_REF)
        print(f"Left boundary at y={Y_REF}: x={x_left:.2f}")
    else:
        x_left = None
        print("Left boundary detection FAILED")
    #---------------------------------------------------------------------------------------------------------------------------#
    ransac_img = img.copy()
    if inliers is not None:
        for i, (x, y) in enumerate(left_edge_points):

            if inliers[i]:
                # Green = trusted
                color = (0, 255, 0)
            else:
                # Red = rejected
                color = (0, 0, 255)

            cv.circle(ransac_img, (x, y), 1, color, -1)

    if left_curve is not None:
        y_values = np.arange(45, 120) # Starts from 45 because this is the beginning of our ROI mask
        x_values = np.polyval(left_curve, y_values)

        # Only draw positions that are actually inside the image
        valid = ((x_values >= 0) & (x_values < img.shape[1]))

        curve_points = np.column_stack((x_values[valid], y_values[valid])).astype(np.int32)

        # Connect curve_points
        cv.polylines(ransac_img, [curve_points.reshape(-1, 1, 2)], False, (255, 0, 0), 1)


    plot_images(img, filtered_left_mask, ransac_img,
                titles=[f"original (alphs: {alpha})", "filtered left mask", "RANSAC result"])


def validate_left_boundary(edge_points, inliers):
    if inliers is None:
        return False

    points = np.array(edge_points, dtype=np.float64)
    inlier_points = points[inliers]

    inlier_ys = inlier_points[:, 1]
    y_min = np.min(inlier_ys)
    y_max = np.max(inlier_ys)

    y_span = y_max - y_min

    print(f"y_min: {y_min}, y_max: {y_max}")

    if y_span < MIN_LEFT_Y_SPAN:
        return False

    return True


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
    num_labels, labels, stats, centroids = \
        cv.connectedComponentsWithStats(left_mask, connectivity=8)

    filtered_mask = np.zeros_like(left_mask)
    accepted_labels = []

    for i in range(1, num_labels):

        area = stats[i, cv.CC_STAT_AREA]

        if area < min_area:
            continue

        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]

        aspect_ratio = w / h

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