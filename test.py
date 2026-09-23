import cv2 as cv
import sys
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
    img = cv.imread("imgs/850_cam_image_array_.jpg")

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

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, GUASSIAN_KERNAL, 0)

    components = get_components(left_candidates, 3)
    cc_img = draw_components(img, components)

    plot_images(img, left_candidates, cc_img, 
                titles=[f"original (alphs: {alpha})", "left candidates", "components"])


def get_components(left_mask, min_area):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(left_mask, connectivity=8)

    components = []

    for i in range(1, num_labels):

        area = stats[i, cv.CC_STAT_AREA]

        if area < min_area:
            continue

        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]

        aspect_ratio = w / h
        fill_ratio = area / (w * h)

        cx, cy = centroids[i]

        print(
            f"#{i}: "
            f"area={area}, "
            f"w={w}, h={h}, "
            f"aspect={aspect_ratio:.2f}, "
            f"fill={fill_ratio:.2f}, "
            f"centroid=({cx:.1f}, {cy:.1f})"
        )

        components.append({
            "label": i,
            "area": area,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy)
        })

    return components


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