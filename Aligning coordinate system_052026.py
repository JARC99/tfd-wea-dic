import glob
import math
import os
from multiprocessing import Process, Queue, Semaphore

import easygui
import numpy as np
import pandas as pd
from geomfitty import fit3d, geom3d
from matplotlib import pyplot as plt
from scipy.io import loadmat

from coordsysalign.multiprocessing_fns import (
    SharedMemory,
    process_out_files,
    put_to_queue,
    read_file,
    read_file_loop,
    read_file_mean_aoi_pos_2d,
    read_file_point_loop,
    read_file_pos_2d_at_index,
)
from coordsysalign.transformation_fns import (
    calculate_circle_rotation_matrix,
    find_x_rotation_matrix,
)

# ----------------------------------------------------------------------------------------------------------------------
# Define input values.
# ----------------------------------------------------------------------------------------------------------------------

# Set the values of the boolean flags used to control the program flow.
SAVE_OUTPUT_FLAG = True
SUBSET_FLAG = True  # TODO: Implement a prompt that lets you specify the number of files you want to use.

YAW_ANGLE_FLAG = False
FRAME_0_REF_FLAG = False

# Specify the number of processors used to read and write on the files.
N_PROCESSES = 16

# Specify the number of marked blades. This number is used to identify the AoIs closest to the rotor hub.
N_MARKED_BLADES = 3

# List the variables that should be stored in the final .csv files. The first column of the file will always contain
# the index.
variables_export_name_out_file = [
    "X",
    "Y",
    "Z",
    "U",
    "V",
    "W",
    "SIGMA_X",
    "SIGMA_Y",
    "SIGMA_Z",
    "sigma",
]

# List the column titles to be used in the .csv files.
variables_export_name_csv_file = [
    '"Index [1]"',
    '"X [mm]"',
    '"Y [mm]"',
    '"Z [mm]"',
    '"U [mm]"',
    '"V [mm]"',
    '"W [mm]"',
    '"Sigma_X [mm]"',
    '"Sigma_Y [mm]"',
    '"Sigma_Z [mm]"',
    '"sigma [pixel]"',
]

if __name__ == "__main__":
    out_file_dir = easygui.diropenbox("Select the folder containing the .out-files")
    print("Directory containing the .out files: ", out_file_dir)
    out_file_list = sorted(glob.glob(out_file_dir + "/*.out"))

    if SUBSET_FLAG:
        subset_size = input(
            "Specify how many files should be considered (must be <= {0}): ".format(
                len(out_file_list)
            )
        )
        out_file_list = out_file_list[: int(subset_size)]
    else:
        pass

    n_frames = len(out_file_list)

    # Load and compute the information needed to use the yaw angle time series
    if YAW_ANGLE_FLAG:
        yaw_angle_file = easygui.fileopenbox("Select the Yaw Angle Time-Series File")

        yaw_angle_array = np.array(loadmat(yaw_angle_file)["yaw_wea"])[
            :n_frames
        ].flatten()

    else:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Find suitable measurement points using the information from the first frame.
    # -------------------------------------------------------------------------------------------------------------------

    print("\r", "Step 1/5: Search for suitable measurement points...", end="")

    # Read the first .out file from the folder. The returned arrays contain information of all the points in that first
    # frame.
    found_array_f0, coordinates_f0, xyz_sigmas_f0, aoi_number_f0, index_in_aoi_f0 = (
        read_file(out_file_list[0], None)
    )

    # Extract the indexes of the points that are visible in the first frame.
    found_indices_f0 = np.nonzero(found_array_f0 == 1)[0]

    # Fill the index list with 1 out of 10 of the visible points and then turn it into an array.
    test_subsets_list = []
    for i in range(0, len(found_indices_f0), 10):
        test_subsets_list.append(found_indices_f0[i])
    test_subsets_list = np.array(test_subsets_list)

    # Initialize the class instances needed for the parallel processing of the .out files multiprocessing package. The
    # previously found test_subsets_list is specified so not all the points in the file need to be read.
    shared_mem = SharedMemory(
        [
            found_array_f0[test_subsets_list],
            coordinates_f0[test_subsets_list],
            xyz_sigmas_f0[test_subsets_list],
            aoi_number_f0[test_subsets_list],
        ]
    )

    file_path_queue = Queue(maxsize=30)

    put_to_queue_process = Process(
        target=put_to_queue, args=(out_file_list, file_path_queue, N_PROCESSES)
    )
    put_to_queue_process.start()

    workers = []
    for _ in range(N_PROCESSES):
        worker = Process(
            target=read_file_loop, args=(file_path_queue, shared_mem, test_subsets_list)
        )
        worker.start()
        workers.append(worker)

    # Pre-allocate arrays to store the cumulative sum information of each of the subset points throughout all the
    # available frames.
    visible_counter = np.ones(len(test_subsets_list), int)
    sigmas = np.zeros(len(test_subsets_list), float)
    distance = np.ones(len(test_subsets_list), float)

    # Define the initial value of the tracker variables for the first iteration.
    last_coordinates = None
    found_last_time = None

    # Extract the data contained in each .out file and update the cumulative sums for visibility,
    # uncertainty (sigmas), and distance.
    file_counter = 0
    for out_file in out_file_list:
        found_array, coordinates, xyz_sigmas, aoi_number = shared_mem.get()

        # Count how many times each point was visible throughout the measurement.
        visible_counter = visible_counter + found_array

        # Determine the indices of the points that are visible for the current frame
        indices_found = np.nonzero(found_array == 1)[0]

        # Add the uncertainty values for the visible points for each frame.
        sigmas[indices_found] = sigmas[indices_found] + np.linalg.norm(
            xyz_sigmas[indices_found], axis=1
        )

        # Capture the edge case for the first file.
        if last_coordinates is None:
            last_coordinates = coordinates
            found_last_time = found_array
            continue

        else:
            # Find the indexes of the points that are visible in both the current and the previous file.
            indices_found = np.nonzero((found_array + found_last_time) == 2)[0]

            # Compute the distance traveled between the last and the current frame
            distance[indices_found] = (
                distance + np.linalg.norm(last_coordinates - coordinates, axis=1)
            )[indices_found]

            # Prepare the last_coordinates and found_last_time arrays for the next loop of calculations
            last_coordinates = coordinates.copy()
            found_last_time = found_array

        file_counter = file_counter + 1

        print(
            "\r",
            "Step 1/5: Search for suitable measurement points... ",
            int((file_counter / len(out_file_list)) * 100),
            "%",
            end="",
        )

    # Wait until all the individual reading processes are finished
    put_to_queue_process.join()
    for worker in workers:
        worker.join()

    print("\r", "Step 1/5: Search for suitable measurement points...  100 %")

    # ------------------------------------------------------------------------------------------------------------------
    # Compute the needed AoI arrays (AoI number, order, Blade, etc)
    # ------------------------------------------------------------------------------------------------------------------

    # Find the index of the point for which the product of the distance per frame and the uncertainty per frame is
    # the smallest (i.e. the point with the least movement). An additional soft constraint is added so the point to be
    # visible at least 90% of the time.
    index_least_movement = np.argmin(
        (distance / visible_counter) * (sigmas / visible_counter)
        + np.where(
            visible_counter >= np.max(visible_counter) * 0.9, 1, np.max(distance) * 1000
        )
    )

    # Find the index of the point for which the product of the inverse of the distance per frame and the uncertainty
    # per frame is the smallest (i.e. the point with the most movement). An additional soft constraint is added so the
    # point to be visible at least 90% of the time.
    index_most_movement = np.argmin(
        (1000 / (distance / visible_counter)) * (sigmas / visible_counter)
        + np.where(
            visible_counter >= np.max(visible_counter) * 0.9, 1, np.max(distance) * 1000
        )
    )

    # Determine the number of unique AoIs in the WEA
    available_aoi_ids = np.unique(aoi_number_f0)

    # Find a single point for each AoI that can be later used to measure the deformation of the
    # blade. Similarly to the previous case, this point should be visible most of the time and its uncertainty should
    # be low.
    interesting_subsets = []
    id_counter = 0
    for local_aoi_id in available_aoi_ids:
        # Find the indexes of the points that are located within the current AoI.
        local_sub_ids = np.nonzero(aoi_number_f0[test_subsets_list] == local_aoi_id)[0]

        # Find index of the point within the current AoI which has the smallest uncertainty and is visible at least
        # 90% of the time.
        id_of_good_subset = id_counter + np.argmin(
            (sigmas[local_sub_ids] / visible_counter[local_sub_ids])
            + np.where(
                visible_counter[local_sub_ids]
                >= np.max(visible_counter[local_sub_ids]) * 0.9,
                1,
                np.max(distance[local_sub_ids]) * 1000,
            )
        )

        # Append the found index to the interesting_subsets list.
        interesting_subsets.append(id_of_good_subset)

        # Take into consideration the offset for the computation of the next AoIs.
        id_counter += len(local_sub_ids)

    interesting_subsets = np.array(interesting_subsets)

    # Compute the average speed of each of the analyzed points.
    distance_per_frame = distance / visible_counter

    # Filter the data from the first frame so that only the indexes of the test_subset_list are considered
    found_array_f0 = found_array_f0[test_subsets_list]
    coordinates_f0 = coordinates_f0[test_subsets_list]
    xyz_sigmas_f0 = xyz_sigmas_f0[test_subsets_list]
    aoi_number_f0 = aoi_number_f0[test_subsets_list]
    index_in_aoi_f0 = index_in_aoi_f0[test_subsets_list]
    found_indices_f0 = np.nonzero(found_array_f0 == 1)[0]

    # Compute the average speed and position of every AoI. This information will be used to determine where on the
    # rotor the AoI is located.
    mean_speed_array = np.empty((len(available_aoi_ids)))
    mean_position_array = np.empty((len(available_aoi_ids), 3))
    for aoi_index in range(len(available_aoi_ids)):
        # Get the indices of the points located within the current AoI and are visible.
        indices_for_aoi = np.nonzero(aoi_number_f0 == aoi_index)[0]

        # Extract the speeds and position of the relevant points.
        distance_per_frame_aoi = distance_per_frame[indices_for_aoi]
        positions_in_aoi = coordinates_f0[indices_for_aoi]

        # Compute the average speed and position of the AoI.
        mean_speed_aoi = np.mean(distance_per_frame_aoi)
        mean_position_aoi = np.mean(positions_in_aoi, axis=0)

        # Add the computed mean speed nd position values to the corresponding array for the current AoI.
        mean_speed_array[aoi_index] = mean_speed_aoi
        mean_position_array[aoi_index] = mean_position_aoi

    # Get the indexes of the AoIs with the smallest speed values.
    aoi_ids_near_center = np.argsort(mean_speed_array)[:N_MARKED_BLADES]

    # Determine which AoI belongs to blade A using its averaged coordinates.
    idx_bladeA = np.argmax(mean_position_array[aoi_ids_near_center, 1])
    idx_bladeB = np.argmin(mean_position_array[aoi_ids_near_center, 0])
    idx_bladeC = np.argmax(mean_position_array[aoi_ids_near_center, 0])

    # Re-arrange the array based on the found indices.
    aoi_ids_near_center = aoi_ids_near_center[[idx_bladeA, idx_bladeB, idx_bladeC]]

    # Store all the points that belong to the innermost AoIs (this will be used for visualization).
    indices_of_inner_subsets = np.array([], dtype=int)
    for aoi_id in aoi_ids_near_center:
        indices_of_inner_subsets = np.append(
            indices_of_inner_subsets, np.nonzero(aoi_number_f0 == aoi_id)[0]
        )

    # Determine the blade each AoI belongs to. For each AoI the distances between its mean location and
    # those of the AoIs closest to the rotor hub are computed. The smallest distance gives away which blade the AoI
    # belongs to.
    blade_number_of_aoi = np.empty((len(available_aoi_ids)))
    for aoi_id in range(len(available_aoi_ids)):
        blade_number_of_aoi[aoi_id] = np.argmin(
            np.linalg.norm(
                mean_position_array[aoi_id] - mean_position_array[aoi_ids_near_center],
                axis=1,
            )
        )

    # Create a list that contains the position of each of the AoIs within their corresponding rotor blade. The
    # innermost AoI always has the index 0.
    aoi_to_blade_aoi = np.array([])
    for aoi_id in range(len(available_aoi_ids)):
        blade_id = blade_number_of_aoi[aoi_id]
        aoi_to_blade_aoi = np.append(
            aoi_to_blade_aoi,
            np.searchsorted(
                np.sort(mean_speed_array[np.nonzero(blade_number_of_aoi == blade_id)]),
                mean_speed_array[aoi_id],
            ),
        )

    # Plot a diagram with the obtained information.
    blade_name_list = ["A", "B", "C"]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.grid()
    ax.scatter(
        coordinates_f0.T[0],
        coordinates_f0.T[1],
        coordinates_f0.T[2],
        c="b",
        alpha=0.25,
        s=10,
    )
    ax.scatter(
        coordinates_f0[index_least_movement].T[0],
        coordinates_f0[index_least_movement].T[1],
        coordinates_f0[index_least_movement].T[2],
        c="g",
        s=180,
        label="Rotor Path Computation Point",
    )
    ax.scatter(
        coordinates_f0[index_most_movement].T[0],
        coordinates_f0[index_most_movement].T[1],
        coordinates_f0[index_most_movement].T[2],
        c="r",
        s=80,
        label="Blade Tips",
    )
    ax.scatter(
        coordinates_f0[indices_of_inner_subsets].T[0],
        coordinates_f0[indices_of_inner_subsets].T[1],
        coordinates_f0[indices_of_inner_subsets].T[2],
        c="y",
        s=30,
        label="Blade Roots",
    )
    ax.scatter(
        coordinates_f0[interesting_subsets].T[0],
        coordinates_f0[interesting_subsets].T[1],
        coordinates_f0[interesting_subsets].T[2],
        c="m",
        s=80,
        label="Good Points",
    )
    ax.scatter(
        mean_position_array.T[0],
        mean_position_array.T[1],
        mean_position_array.T[2],
        c="cyan",
        s=80,
        label="AoI Average Location",
    )
    for aoi_id in range(len(available_aoi_ids)):
        ax.text(
            mean_position_array[aoi_id, 0],
            mean_position_array[aoi_id, 1],
            mean_position_array[aoi_id, 2],
            "{0}: {1}".format(
                blade_name_list[int(blade_number_of_aoi[aoi_id])],
                round(aoi_to_blade_aoi[aoi_id]),
            ),
            size=10,
            zorder=1,
            color="k",
        )
    ax.set_aspect("equal")
    ax.set_title("Found Points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax = plt.gca()
    plt.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Compute the reference circles.
    # ------------------------------------------------------------------------------------------------------------------

    print("\r", "Step 2/5: Create circle... ", end="")

    # FRAME 0: Start by using the innermost points to estimate the orientation of the rotor on the first frame by first
    # calculating the radius to the root AoIs.
    hub_aoi_rad = np.mean(
        np.linalg.norm(
            mean_position_array[aoi_ids_near_center]
            - np.mean(coordinates_f0[indices_of_inner_subsets], axis=0),
            axis=1,
        )
        / 1000
    )

    # AVERAGED: Initialize the multiprocessing class instances needed for reading the point-cloud dta to compute the averaged circle.
    shared_mem = SharedMemory([found_array_f0[[0]], coordinates_f0[[0]]])
    file_path_queue = Queue(maxsize=30)

    put_to_queue_process = Process(
        target=put_to_queue, args=(out_file_list, file_path_queue, N_PROCESSES)
    )
    put_to_queue_process.start()

    workers = []
    for _ in range(N_PROCESSES):
        worker = Process(
            target=read_file_point_loop,
            args=(file_path_queue, shared_mem, test_subsets_list[index_least_movement]),
        )
        worker.start()
        workers.append(worker)

    # Pre-allocate an array where the point's coordinates in every frame should be stored. If the chosen point is not
    # visible in a certain frame, nothing is stored.
    coordinates = np.zeros((len(out_file_list), 3), float)
    file_counter = 0
    not_found_counter = 0

    # Store the point's coordinates for each .out file contianed in the folder.
    for out_file in out_file_list:
        found_array, real_points = shared_mem.get()
        if found_array[0] == 1:
            coordinates[file_counter - not_found_counter] = real_points.copy()
        else:
            not_found_counter += 1
        file_counter = file_counter + 1
        print(
            "\r",
            "Step 2/5: Create circle... ",
            int((file_counter / len(out_file_list)) * 100),
            "%",
            end="",
        )

    put_to_queue_process.join()
    for worker in workers:
        worker.join()

    print("\r", "Step 2/5: Create circle...  100 % ")

    print("\r", "Step 3/5: Calculate circle...", end="")

    # FRAME 0: Fit a 3D circle to the points found using the first frame.
    initial_guess = geom3d.Circle3D(
        np.mean(coordinates_f0[indices_of_inner_subsets], axis=0),
        [1, 0, 0],
        hub_aoi_rad,
    )
    circle_f0 = fit3d.circle3D_fit(
        coordinates_f0[indices_of_inner_subsets],
        initial_guess=initial_guess,
    )

    # Store the obtained values in mutable containers.
    circle_center_f0 = circle_f0.center
    circle_direction_f0 = circle_f0.direction

    # AVERAGED: Slice the array to only consider the length of the array with valid coordinate values.
    coordinates = coordinates[0 : file_counter - not_found_counter]

    # Fit a 3D circle to the points found throughout the dataset.
    initial_guess = geom3d.Circle3D(np.mean(coordinates, axis=0), [1, 0, 0], 7)
    circle = fit3d.circle3D_fit(coordinates, initial_guess=initial_guess)
    circle_center = circle.center.copy()
    circle_direction = circle.direction.copy()

    # Verify that the circle's x-axis points in the correct direction. If not, flip the vector.
    if circle_direction[-1] > 0:
        circle_direction = circle_direction * -1

    if circle_direction[-1] * circle_direction_f0[-1] < 0:
        circle_direction_f0 = circle_direction_f0 * -1
    else:
        pass

    # Compute the angular offset between the rotor in the first frame and that of the averaged rotor direction.
    if YAW_ANGLE_FLAG:
        if FRAME_0_REF_FLAG:
            pass
        else:
            delta_theta = np.rad2deg(
                np.arccos(
                    np.dot(circle_f0.direction, circle.direction)
                    / (
                        np.linalg.norm(circle_direction_f0)
                        * np.linalg.norm(circle_direction)
                    )
                )
            )

            if delta_theta > 90:
                delta_theta = delta_theta - 180
            else:
                pass

            yaw_angle_ave = yaw_angle_array[0] - delta_theta
    else:
        pass

    # Plot the computed geometries (i.e. the averaged circle and that of the first frame).
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.grid()
    ax.scatter(coordinates.T[0], coordinates.T[1], coordinates.T[2])
    ax.set_aspect("equal")
    ax.set_title("Average and First Frame Circle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    random = np.array([1, 0, 0])
    u = np.cross(circle.direction, random)
    v = np.cross(circle.direction, u)
    theta = np.linspace(0, 2 * np.pi, 100)

    circle_pts = circle.radius * (
        np.outer(u, np.cos(theta)) + np.outer(v, np.sin(theta))
    )
    circle_pts = circle.center + circle_pts.T

    ax.plot(
        circle.center[0],
        circle.center[1],
        circle.center[2],
        marker="X",
        color="r",
    )
    ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2], color="b")

    ax.quiver(
        circle.center[0],
        circle.center[1],
        circle.center[2],
        circle.direction[0] * circle.radius,
        circle.direction[1] * circle.radius,
        circle.direction[2] * circle.radius,
        color="b",
        label="Average",
    )

    random = np.array([1, 0, 0])
    u0 = np.cross(circle_f0.direction, random)
    v0 = np.cross(circle_f0.direction, u0)
    theta = np.linspace(0, 2 * np.pi, 100)

    circle_pts0 = circle.radius * (
        np.outer(u0, np.cos(theta)) + np.outer(v0, np.sin(theta))
    )
    circle_pts0 = circle_f0.center + circle_pts0.T

    ax.plot(
        circle_f0.center[0],
        circle_f0.center[1],
        circle_f0.center[2],
        marker="D",
        color="black",
    )
    ax.plot(circle_pts0[:, 0], circle_pts0[:, 1], circle_pts0[:, 2], color="m")
    ax.quiver(
        circle_f0.center[0],
        circle_f0.center[1],
        circle_f0.center[2],
        circle_f0.direction[0] * circle.radius,
        circle_f0.direction[1] * circle.radius,
        circle_f0.direction[2] * circle.radius,
        color="m",
        label="Frame 0",
    )

    ax.legend()
    ax = plt.gca()
    fig.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Calculate needed rotation matrices and compute the corresponding transformations
    # ------------------------------------------------------------------------------------------------------------------

    # Calculate the needed rotation matrix to bring the rotor into the y-z plane based on the averaged circle.
    rotation_matrix = calculate_circle_rotation_matrix(circle_direction, 1)

    # Center and rotate the found rotor points so the rotor is on the y-z plane.
    coordinates_temp = np.dot(rotation_matrix, (coordinates - circle_center).T).T

    # Verify that the obtained rotation is in the correct direction.
    direction_array = np.zeros((len(coordinates_temp) - 1), np.int8)
    for i in range(len(coordinates_temp) - 1):
        pt = coordinates_temp[i]
        v0 = pt - coordinates_temp[i - 1]
        c = np.cross(pt, v0)

    if c[0] < 0:
        direction_array[i] = -1
    else:
        direction_array[i] = 1

    if np.mean(direction_array > 0):
        x_axis_alignment = 1
    else:
        x_axis_alignment = -1

    # Invert the x-axis if the rotation direction was found to be incorrect.
    rotation_matrix = calculate_circle_rotation_matrix(
        circle_direction, x_axis_alignment
    )

    # Calculate the needed rotation around the x-axis, so that Blade A is always at 12:00.
    most_moved_point = np.dot(
        rotation_matrix, (coordinates_f0[index_most_movement] - circle_center).T
    ).T
    most_moved_point_blade = blade_number_of_aoi[aoi_number_f0[index_most_movement]]

    diff = most_moved_point / np.linalg.norm(
        most_moved_point
    )  # TODO: maybe call function instead of repeating this code section.
    x0 = diff[2]
    x1 = diff[1]
    if x0 > 0:
        angle = math.degrees(math.asin(x1))
    else:
        angle = 360 - math.degrees(math.asin(x1))
        angle += 180
    angle += -120 * most_moved_point_blade

    x_rot_angle = math.radians(angle)
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(x_rot_angle), -math.sin(x_rot_angle)],
            [0, math.sin(x_rot_angle), math.cos(x_rot_angle)],
        ]
    )

    rotation_matrix_ave = np.matmul(rot_x, rotation_matrix)

    # Compute the rotations matrices to correct the yaw rotation for each of the frames depending on whether or the
    # YAW_ANGLE_FLAG is active or not. The reference angle can be either the first frame or the average circle.
    rotation_matrix_list = []
    if YAW_ANGLE_FLAG:
        if FRAME_0_REF_FLAG:
            pass
        else:
            z_rot_angle_array = np.deg2rad(-(yaw_angle_ave - yaw_angle_array))

            for z_rot_angle in z_rot_angle_array:
                rot_z = np.array(
                    [
                        [np.cos(z_rot_angle), -np.sin(z_rot_angle), 0],
                        [np.sin(z_rot_angle), np.cos(z_rot_angle), 0],
                        [0, 0, 1],
                    ]
                )
                rotation_matrix_list.append(
                    np.matmul(rot_x, np.matmul(rot_z, rotation_matrix))
                )
    else:
        rotation_matrix_list = [np.matmul(rot_x, rotation_matrix)] * len(out_file_list)

    rotation_matrix_f0 = rotation_matrix_list[0]

    # Rotate the points obtained from the first frame to visualize them.
    coordinates = np.dot(rotation_matrix_ave, (coordinates - circle_center).T).T

    print("\r", "Step 3/5: Calculate circle...  100 %")

    # Plot the reference circles after applying the coordinate transformations.
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.grid()
    ax.scatter(
        coordinates.T[0], coordinates.T[1], coordinates.T[2], label="Point Cloud"
    )

    if YAW_ANGLE_FLAG:
        circle_pts0_rot_w_yaw = np.dot(
            rotation_matrix_f0, (circle_pts0 - circle_f0.center).T
        ).T
        circle_pts0_rot_wo_yaw = np.dot(
            rotation_matrix_ave, (circle_pts0 - circle_f0.center).T
        ).T

        ax.plot(
            circle_pts0_rot_w_yaw[:, 0],
            circle_pts0_rot_w_yaw[:, 1],
            circle_pts0_rot_w_yaw[:, 2],
            color="m",
            label="Frame 0 w/ Yaw Correction",
        )
        ax.plot(
            circle_pts0_rot_wo_yaw[:, 0],
            circle_pts0_rot_wo_yaw[:, 1],
            circle_pts0_rot_wo_yaw[:, 2],
            color="c",
            label="Frame 0 w/o Yaw Correction",
        )

    else:
        pass

    ax.set_title("Circular Path Used for the Coordinate Transformation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax = plt.gca()
    ax.legend()
    ax.set_aspect("equal")
    fig.show()
    plt.pause(0)

    # ------------------------------------------------------------------------------------------------------------------
    # Find a pait of points for each AoI for the torsion calculation
    # ------------------------------------------------------------------------------------------------------------------

    print(
        "\r",
        "Step 4/5: Search points for the rotor blade torsion calculation...",
        end="",
    )

    # Center and rotate the coordinate points of all the points read from the first .out file.
    coordinates_f0 = np.dot(rotation_matrix_f0, (coordinates_f0 - circle_center_f0).T).T

    # Rotate the coordinates of the AoIs mean position using the first frame rotation matrix.
    mean_position_array = np.dot(
        rotation_matrix_f0, (mean_position_array - circle_center).T
    ).T

    # Create an 3D array where you will store the rotation matrices around the x-axis for each individual AoI.
    rotation_matrix_for_aoi = np.empty((len(available_aoi_ids), 3, 3), float)
    for aoi_id in range(len(available_aoi_ids)):
        index_highest_point_on_blade = np.nonzero(
            (blade_number_of_aoi == blade_number_of_aoi[aoi_id])
            & (aoi_to_blade_aoi == np.max(aoi_to_blade_aoi))
        )[0][0]
        rotation_matrix_for_aoi[aoi_id] = find_x_rotation_matrix(
            mean_position_array[index_highest_point_on_blade]
        )

    # Extract the information of each AoI and determine what points are the best to calculate the blade torsion.
    indices_for_aoi_inter_org = np.arange(len(coordinates_f0))
    index_array = np.empty((len(available_aoi_ids), 2), int)
    for aoi_index in range(len(available_aoi_ids)):
        current_best_value = float("inf")
        best_points = None

        indices_for_aoi = np.nonzero(aoi_number_f0 == aoi_index)[0]
        local_coordinates_for_aoi = coordinates_f0[indices_for_aoi]
        local_coordinates_for_aoi = np.dot(
            rotation_matrix_for_aoi[aoi_index], local_coordinates_for_aoi.T
        ).T
        local_sigmas = xyz_sigmas_f0[indices_for_aoi]
        local_visible_counter = visible_counter[indices_for_aoi]
        indices_for_aoi_local = indices_for_aoi_inter_org[indices_for_aoi]
        norm_result = np.linalg.norm(local_sigmas, axis=1) + 1
        k = 0
        for i in range(len(local_coordinates_for_aoi)):
            for j in range(i):
                p1 = local_coordinates_for_aoi[i]
                p2 = local_coordinates_for_aoi[j]

                # TODO: Multiplikation zu + aendern und irgendwie einbauen
                value = (
                    ((abs(p1[2] - p2[2]) + 100) / (abs(p1[1] - p2[1]) + 1))
                    * (norm_result[i] + norm_result[j])
                    * (
                        (np.max(local_visible_counter) + np.max(local_visible_counter))
                        / (local_visible_counter[i] + local_visible_counter[j])
                    )
                )

                if value < current_best_value:
                    current_best_value = value
                    best_points = np.array(
                        [indices_for_aoi_local[i], indices_for_aoi_local[j]]
                    )

        index_array[aoi_index] = best_points
        print(
            "\r",
            "Step 4/5: Search points for rotor blade torsion calculation...",
            int((aoi_index / len(available_aoi_ids)) * 100),
            "%",
            end="",
        )

    print("\r", "Step 4/5: Search points for rotor blade torsion calculation...  100 %")

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.grid()
    ax.scatter(
        coordinates_f0[found_indices_f0].T[0],
        coordinates_f0[found_indices_f0].T[1],
        coordinates_f0[found_indices_f0].T[2],
        c="b",
        alpha=0.25,
        s=10,
    )
    ax.scatter(
        coordinates_f0[index_array].T[0],
        coordinates_f0[index_array].T[1],
        coordinates_f0[index_array].T[2],
        c="r",
        s=80,
        label="Good Points for the torsion calculations",
    )
    ax.set_aspect("equal")
    ax.set_title("Transformed Coordinate System")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim((-60000, 60000))
    plt.legend()
    fig.show()
    plt.pause(0)

    # ------------------------------------------------------------------------------------------------------------------
    # Save the transformed .out files into the previously created folders.
    # ------------------------------------------------------------------------------------------------------------------

    if SAVE_OUTPUT_FLAG:
        # The complete datapoints from the first frame are re-read.
        found_array_f0, coordinates_f0, xyz_sigmas_f0, aoi_number_f0, _ = read_file(
            out_file_list[0], None
        )

        indices_of_inner_subsets = np.array([], dtype=int)
        for aoi_id in aoi_ids_near_center:
            indices_of_inner_subsets = np.append(
                indices_of_inner_subsets, np.nonzero(aoi_number_f0 == aoi_id)[0]
            )

        found_array_first_frame = found_array_f0[indices_of_inner_subsets]
        coordinates_first_frame = coordinates_f0[indices_of_inner_subsets]
        coordinates_first_frame = np.dot(
            rotation_matrix_f0, (coordinates_first_frame - circle_center_f0).T
        ).T

        print("\r", "Step 5/5: Store adjusted measurement points...", end="")

        # Show a 2D image with the named AoIs.
        two_d_coordinates = read_file_mean_aoi_pos_2d(out_file_list[0])
        fig, ax = plt.subplots()
        ax.scatter(
            two_d_coordinates.T[0],
            np.max(two_d_coordinates.T[1]) - two_d_coordinates.T[1],
        )
        plt.title("Found AoI in the Reference Frame (Frame 0)")
        for aoi_id in range(len(two_d_coordinates)):
            annotation = "Blade {0} - AoI: {1}".format(
                blade_name_list[int(blade_number_of_aoi[aoi_id])],
                int(aoi_to_blade_aoi[aoi_id]),
            )

            ax.annotate(
                annotation,
                (
                    two_d_coordinates.T[0][aoi_id],
                    np.max(two_d_coordinates.T[1]) - two_d_coordinates.T[1][aoi_id],
                ),
            )
        plt.show()
        fig.savefig(out_file_dir + "/AoI_Naming.png", dpi=fig.dpi)
        plt.pause(0)

        # If not existent already, create the folders needed to store the processed .out files.
        if not os.path.isdir(out_file_dir + "/koordNachGL/"):
            os.mkdir(out_file_dir + "/koordNachGL/")
        if not os.path.isdir(out_file_dir + "/koordNachGL_noRot/"):
            os.mkdir(out_file_dir + "/koordNachGL_noRot/")
        if not os.path.isdir(out_file_dir + "/SchlagSchwenk/"):
            os.mkdir(out_file_dir + "/SchlagSchwenk/")
        if not os.path.isdir(out_file_dir + "/Torsion/"):
            os.mkdir(out_file_dir + "/Torsion/")

        output_path1 = out_file_dir + "/koordNachGL/"
        output_path2 = out_file_dir + "/koordNachGL_noRot/"
        output_path3 = out_file_dir + "/SchlagSchwenk/"
        output_path4 = out_file_dir + "/Torsion/"

        # Store for each AoI a single point in a separate in a .csv file.
        interesting_subsets_id_vicpy = np.empty((len(interesting_subsets), 3), int)
        interesting_subsets_id_vicpy[:, 0] = index_in_aoi_f0[interesting_subsets]
        interesting_subsets_id_vicpy[:, 1] = index_in_aoi_f0[index_array[:, 0]]
        interesting_subsets_id_vicpy[:, 2] = index_in_aoi_f0[index_array[:, 1]]

        position_of_interesting_points_2d = read_file_pos_2d_at_index(
            out_file_list[0], interesting_subsets_id_vicpy[:, 0]
        )

        # Initiate the needed multiprocessing instances to process and store the .out files.
        shared_mem = SharedMemory(
            [
                np.empty(
                    (
                        len(interesting_subsets_id_vicpy),
                        3,
                        len(variables_export_name_out_file),
                    ),
                    np.float32,
                )
            ]
        )
        file_path_queue = Queue(maxsize=30)

        put_to_queue_process = Process(
            target=put_to_queue, args=(out_file_list, file_path_queue, N_PROCESSES)
        )
        put_to_queue_process.start()

        # Create Semaphore object to monitor the processing progress.
        semaphore = Semaphore(0)

        workers = []
        for _ in range(N_PROCESSES):
            worker = Process(
                target=process_out_files,
                args=(
                    file_path_queue,
                    output_path1,
                    output_path2,
                    -circle_center,
                    rotation_matrix_list,
                    aoi_ids_near_center,
                    found_array_first_frame,
                    coordinates_first_frame,
                    shared_mem,
                    interesting_subsets_id_vicpy,
                    variables_export_name_out_file,
                ),
            )
            worker.start()
            workers.append(worker)

        # The points in the interesting_points subset are brought to the 12:00 position and stored in .csv files.
        # TODO: The uncertainty calculation still needs to be updted.
        good_points_data = np.empty(
            (
                len(out_file_list),
                len(available_aoi_ids),
                len(variables_export_name_out_file),
            ),
            float,
        )
        good_points_torsion = np.empty(
            (
                len(out_file_list),
                len(available_aoi_ids),
                2,
                len(variables_export_name_out_file),
            ),
            float,
        )
        for file_counter in range(len(out_file_list)):
            data = shared_mem.get()
            data = np.array(data)[0]

            for aoi_id in range(len(available_aoi_ids)):
                data[aoi_id, 0, 0:3] = np.dot(
                    rotation_matrix_for_aoi[aoi_id], data[aoi_id, 0, 0:3].T
                ).T
                data[aoi_id, 0, 3:6] = np.dot(
                    rotation_matrix_for_aoi[aoi_id], data[aoi_id, 0, 3:6].T
                ).T

            good_points_data[file_counter] = data[:, 0]
            good_points_torsion[file_counter] = data[:, 1:]
            print(
                "\r",
                "Step 5/5: Store adjusted measurement points... ",
                int((file_counter / len(out_file_list)) * 100),
                "%",
                end="",
            )

        put_to_queue_process.join()
        for worker in workers:
            worker.join()

        # Store the points in .csv files.
        for aoi_id in range(len(available_aoi_ids)):
            dic, dict1, dict2 = {}, {}, {}
            for var_id in range(1, len(variables_export_name_csv_file)):
                dic[variables_export_name_csv_file[var_id]] = good_points_data[
                    :, aoi_id, var_id - 1
                ]
                dict1[variables_export_name_csv_file[var_id]] = good_points_torsion[
                    :, aoi_id, 0, var_id - 1
                ]
                dict2[variables_export_name_csv_file[var_id]] = good_points_torsion[
                    :, aoi_id, 1, var_id - 1
                ]
            df = pd.DataFrame(dic)
            df_1 = pd.DataFrame(dict1)
            df_2 = pd.DataFrame(dict2)

            csv_filename1 = (
                output_path3
                + "/Blade_"
                + str(int(blade_number_of_aoi[aoi_id]))
                + "_AOI_"
                + str(int(aoi_to_blade_aoi[aoi_id]))
                + "_"
                + str(position_of_interesting_points_2d[aoi_id])
                + ".csv"
            )
            csv_filename2 = (
                output_path4
                + "/Blade_"
                + str(int(blade_number_of_aoi[aoi_id]))
                + "_AOI_"
                + str(int(aoi_to_blade_aoi[aoi_id]))
                + "_P1.csv"
            )
            csv_filename3 = (
                output_path4
                + "/Blade_"
                + str(int(blade_number_of_aoi[aoi_id]))
                + "_AOI_"
                + str(int(aoi_to_blade_aoi[aoi_id]))
                + "_P2.csv"
            )
            header_text = (
                '"B'
                + str(int(blade_number_of_aoi[aoi_id]))
                + " AOI"
                + str(int(aoi_to_blade_aoi[aoi_id]))
                + '"'
                + ";" * len(variables_export_name_out_file)
            )

            for csv_file_name in [csv_filename1, csv_filename2, csv_filename3]:
                with open(csv_file_name, "w") as f:
                    f.write(header_text + "\n")
            df.to_csv(
                csv_filename1,
                index_label=variables_export_name_csv_file[0],
                mode="a",
                index=True,
                quotechar="'",
                sep=";",
                decimal=",",
            )
            df_1.to_csv(
                csv_filename2,
                index_label=variables_export_name_csv_file[0],
                mode="a",
                index=True,
                quotechar="'",
                sep=";",
                decimal=",",
            )
            df_2.to_csv(
                csv_filename3,
                index_label=variables_export_name_csv_file[0],
                mode="a",
                index=True,
                quotechar="'",
                sep=";",
                decimal=",",
            )

        print("\r", "Step 5/5: Store adjusted measurement points...  100 %")

        exit()
