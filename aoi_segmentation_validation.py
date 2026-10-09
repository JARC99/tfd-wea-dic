import shutil
from pathlib import Path
from tkinter import filedialog
import matplotlib.pyplot as plt

import cv2
import numpy as np

import tqdm

BLACK_THRESHOLD = 10
MIN_EDGE_PROP = 0.0025
PLOT_FLAG = False

img_path = img_path = Path(filedialog.askdirectory(title="Select Image Folder"))

failed_dir = img_path / "failed_masks"
failed_dir.mkdir(exist_ok=True)

img_list = list(img_path.glob("*.tif"))
fail_count = 0
for img_file in tqdm.tqdm(img_list):

    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {img}")

    total_pixels = img.size

    # Find the pixels that are not black.
    foreground_mask = img > BLACK_THRESHOLD
    foreground_pixel_count = np.sum(foreground_mask)

    # Detect all edges within the visible region in the image using the Canny algorithm.
    raw_edges = cv2.Canny(img, threshold1=30, threshold2=100)
    masked_edges = cv2.bitwise_and(raw_edges, foreground_mask.astype("uint8"))

    # Compute the edge proportion in the visible part of the image
    edge_pixel_count = np.sum(masked_edges > 0)
    edge_prop = edge_pixel_count / max(foreground_pixel_count, 1)

    # Evaluate the proportions
    if edge_prop >= MIN_EDGE_PROP:
        status = "PASSED"
    else:
        status = "FAILED"

        fail_count += 1
        shutil.copy(img_file, failed_dir / img_file.name)

    if PLOT_FLAG:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(img, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis("off")

        ax2.imshow(raw_edges, cmap='gray')
        ax2.set_title("Canny Edges")
        ax2.axis("off")

        fig.suptitle(f"Mask Status: {status}\n Edge Percentage: {edge_prop*100:.2f} %")
        plt.tight_layout()
        fig.show()
    else:
        pass

