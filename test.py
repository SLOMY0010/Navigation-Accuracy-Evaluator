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
GUASSIAN_KERNAL = (3, 3)


def main():
    img = cv.imread("imgs/437_cam_image_array_.jpg")

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

    img_points = img.copy()
    for x, y, in left_edge_points:
        cv.circle(img_points, (x, y), 1, (0, 0, 255), -1)

    plot_images(img, img_points, filtered_left_mask, 
                titles=[f"original (alphs: {alpha})", "Points on original", "filtered left mask"])


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
    blur = cv.GaussianBlur(gray, GUASSIAN_KERNAL, 0)
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