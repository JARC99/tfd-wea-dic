import shutil
from multiprocessing import Pool
from pathlib import Path
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def count_visible_aois(img_file):
    # Create the folder to store the failed images
    failed_path = img_file.parent / "failed_masks"
    failed_path.mkdir(exist_ok=True)

    # Load the image and convert it to grayscale and increase contrast
    img_raw = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img_hcont = cv2.convertScaleAbs(img_raw, alpha=7.0)

    # Find the pixels within the originally applied mask
    mask_og = (img_hcont > 0).astype(np.uint8) * 255

    # Shrink the original mask to avoid capturing its original edges
    shrink_kernel = np.ones((25, 25), np.uint8)
    mask_shrunk = cv2.erode(mask_og, shrink_kernel, iterations=2)

    # Find the center of the center of the original mask to eliminate the centra section of the WEA using a circle
    moments_mask_og = cv2.moments(mask_og)

    if moments_mask_og["m00"] > 0:
        width_mask_og = int(moments_mask_og["m10"] / moments_mask_og["m00"])
        height_mask_og = int(moments_mask_og["m01"] / moments_mask_og["m00"])
    else:
        width_mask_og, height_mask_og = mask_og.shape

    cv2.circle(mask_shrunk, (width_mask_og, height_mask_og), int(0.35 * height_mask_og), 0, thickness=-1)

    # Fill the background of the image with the median value of the shown section to avoid jumps in the next clculations
    img_raw_med = np.median(img_hcont[img_hcont > 0])
    img_med_val = img_hcont.copy()
    img_med_val[mask_shrunk == 0] = int(img_raw_med)

    # Calculate the local standard deviation of each pixel using a 3x3 kernel, then normallize to 8-bit values
    float_img = img_med_val.astype(np.float32)
    mean = cv2.blur(float_img, (3, 3))
    sq_mean = cv2.blur(float_img ** 2, (3, 3))
    local_var = sq_mean - mean ** 2

    img_sigma = np.sqrt(np.maximum(local_var, 0))
    img_sigma = cv2.normalize(
        img_sigma, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Find the Otsu threshold to divide the image between high standard deviation and low standard deviation sections, then apply it to the image
    otsu_thresh, _ = cv2.threshold(
        img_sigma[mask_shrunk > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, img_thresh = cv2.threshold(img_sigma, otsu_thresh, 255, cv2.THRESH_BINARY)

    # Clean up noise using morphological operations
    kernel_thresh_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel_thresh_close)

    kernel_thresh_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_OPEN, kernel_thresh_open)

    img_final = img_morph  # cv2.bitwise_and(img_morph, img_morph, mask=mask_shrunk)

    # Save mask
    cv2.imwrite("turbine_segmented.png", img_final)

    # Find all the connected components in the final binary mask
    n_regions, _, stats, centroids = cv2.connectedComponentsWithStats(
        img_final
    )

    # Specify a minimum size in pixels to consider a region a valid AoI
    MIN_REGION_AREA = 500  # Minimum pixel count to be considered a valid part
    n_found_aoi = 0
    aoi_centroid_list = []
    for i in range(1, n_regions):  # Start at 1 to skip background with index 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_REGION_AREA:
            n_found_aoi += 1
            aoi_centroid_list.append(centroids[i])

    # print(f"\n Number of found regions larger than {MIN_REGION_AREA} pixels in {img_file.name}: {n_found_aoi}")

    # Plot images
    if False:
        titles = [
            "1. Original Image",
            "2. Original Mask w/ Circle",
            "3. Background-Filled Image",
            "4. Standard Deviation",
            "5. Otsu Threshold",
            "6. Final Mask",
        ]
        images = [
            img_raw,
            mask_shrunk,
            img_med_val,
            img_sigma,
            img_thresh,
            img_final,
        ]

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(images[-1], cmap="gray")
        for cx, cy in aoi_centroid_list:
            ax.plot(
                cx,
                cy,
                marker="x",
                color="magenta",
                markersize=4,
                linestyle="None",
            )
        ax.axis("off")

        # ax_list = []
        # for i in range(len(images)):
        #     ax_list.append(fig.add_subplot(2, 3, i + 1))
        #     ax_list[i].imshow(images[i], cmap="gray")
        #     ax_list[i].set_title(titles[i], fontsize=10)
        #     ax_list[i].axis("off")
        #
        # if i == 5:
        #     for cx, cy in aoi_centroid_list:
        #         ax_list[i].plot(
        #             cx,
        #             cy,
        #             marker="x",
        #             color="magenta",
        #             markersize=4,
        #             linestyle="None",
        #         )

        fig.suptitle(img_file.name)
        plt.show()
        plt.close(fig)

    # If less than 12 AoIs are found (the three hub AoIs are covered by the circle) raise a warning and store the image in the failed_masks folder
    if n_found_aoi < 12:
        shutil.copy(img_file, failed_path / img_file.name)
        return 1
    else:
        return 0


if __name__ == "__main__":
    # Determine optimal number of available CPUs to use
    N_PROCESSES = 2  # max(1, cpu_count() // 2)

    # Select the folder containing the images
    folder_path = Path(filedialog.askdirectory(
        title="Select Folder Containing .TIF Images"
    ))

    img_file_list = list(folder_path.glob("*.tif"))

    print(f"Selected folder: {folder_path}")
    print(f"Found {len(img_file_list)} .TIF images to process.")

    # Use a multiprocessing pool to execute the process in parallel
    with Pool(processes=N_PROCESSES) as pool:
        result_list = list(
            tqdm(
                pool.imap(count_visible_aois, img_file_list),
                total=len(img_file_list),
                desc="Processing Turbines",
            )
        )

    print(f"Batch processing complete!\n Failed masks: {sum(result_list)}/{len(result_list)}")
