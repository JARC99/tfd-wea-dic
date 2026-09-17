import math
import os.path

import numpy as np
from scipy.optimize import least_squares
from vicpyx import VicDataSet

from geomfitty import fit3d, geom3d


def quaternion_to_rotation_matrix(Q):
    """
    Calculate quaternions.
    """

    q0 = Q[0]
    qx = Q[1]
    qy = Q[2]
    qz = Q[3]

    r00 = q0 ** 2 + qx ** 2 - qy ** 2 - qz ** 2
    r01 = 2 * (qx * qy - q0 * qz)
    r02 = 2 * (qx * qz + q0 * qy)

    r10 = 2 * (qy * qx + q0 * qz)
    r11 = q0 ** 2 - qx ** 2 + qy ** 2 - qz ** 2
    r12 = 2 * (qy * qz - q0 * qx)

    r20 = 2 * (qz * qx - q0 * qy)
    r21 = 2 * (qz * qy + q0 * qx)
    r22 = q0 ** 2 - qx ** 2 - qy ** 2 + qz ** 2

    rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return rot_matrix


def find_rotation(right_co, left_co):
    """
    Berechnet eine Rotationsmatrix, um die Punkte im left_co-Koordinatensystem nach der Rotation im right_co-Koordinatensystem auszurichten.
    Die Punkte sind nach dem Anwenden der Rotationsmatrix noch nicht gleich, da die Translation und die Skalierung noch berücksichtigt werden muss

        right_co - Punkte im ersten Koordinatensystem (Referenz)
        left_co - Die gleichen Punkte mit den Koordinaten nach dem zweiten Koordinatensystem
    """

    mean_left = np.mean(left_co, axis=0)
    mean_right = np.mean(right_co, axis=0)

    left_co = left_co - mean_left
    right_co = right_co - mean_right

    M = np.empty((3, 3), float)

    for i in range(3):
        for j in range(3):
            M[j, i] = np.dot(right_co[:, i], left_co[:, j].T)
            # M[j, i] = np.dot(right_co[i], left_co[j].T)

    N = np.array(
        [
            [
                M[0, 0] + M[1, 1] + M[2, 2],
                M[1, 2] - M[2, 1],
                M[2, 0] - M[0, 2],
                M[0, 1] - M[1, 0],
            ],
            [
                M[1, 2] - M[2, 1],
                M[0, 0] - M[1, 1] - M[2, 2],
                M[0, 1] + M[1, 0],
                M[2, 0] + M[0, 2],
            ],
            [
                M[2, 0] - M[0, 2],
                M[0, 1] + M[1, 0],
                -M[0, 0] + M[1, 1] - M[2, 2],
                M[1, 2] + M[2, 1],
            ],
            [
                M[0, 1] - M[1, 0],
                M[2, 0] + M[0, 2],
                M[1, 2] + M[2, 1],
                -M[0, 0] - M[1, 1] + M[2, 2],
            ],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(N)
    # eigenvalues, eigenvectors = np.linalg.eig(N) # TODO: uncooment
    eigenvectors = eigenvectors.T
    max_index = np.argmax(eigenvalues)
    test = eigenvectors[max_index] / np.linalg.norm(eigenvectors[max_index])

    return quaternion_to_rotation_matrix(test)


def find_translation(right_co, left_co_temp):
    """
    Berechnet einen Translationsvektor, um Punkte aus dem Koordinatensystem left_co_temp in Punkte nach dem Koordinatensystem right_co zu überführen.
    Die Rotation muss bereits angepasst worden sein

        right_co - Punkte im ersten Koordinatensystem (Referenz)
        left_co_temp - Die gleichen Punkte mit den Koordinaten nach dem zweiten Koordinatensystem
    """

    mean_right = np.mean(right_co, axis=0)
    mean_left = np.mean(left_co_temp, axis=0)
    return mean_right - mean_left


def find_x_rotation_matrix(point):
    """
    Berechnet eine Rotationsmatrix, welche einen Punkt um die x-Achse auf die "12-Uhr" Position dreht

        point - Mit diesem Punkt wird die Rotationsmatrix ermittelt
    """

    diff = point / np.linalg.norm(point)
    x0 = diff[2]
    x1 = diff[1]
    if x0 > 0:
        angle = math.degrees(math.asin(x1))
    else:
        angle = 360 - math.degrees(math.asin(x1))
        angle += 180
    x_rot_angle = math.radians(angle)
    return np.array(
        [
            [1, 0, 0],
            [0, math.cos(x_rot_angle), -math.sin(x_rot_angle)],
            [0, math.sin(x_rot_angle), math.cos(x_rot_angle)],
        ]
    )


def calculate_circle_rotation_matrix(circle_direction, direction):
    """
    Compute the rotation matrix needed to rotate the vector normal to the circle surface to a specified direction
    """

    # Ab hier Rechnung aus der Dissertation um Rotationsmatrix zu bestimmen
    circle_direction = np.array(circle_direction)
    k = (
            (direction * circle_direction) / (np.linalg.norm(circle_direction))
    )  # Manchmal zeigt der Rotor in die falsche Richtung. Dann muss das Vorzeichen von k angepasst werden
    n = np.array([1, 0, 0])

    v = np.cross(k, n)
    s = np.linalg.norm(v)
    c = np.dot(k, n)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # finale Rotationsmatrix, die num immer zum Ausrichten des Koordinatensystems verwendet wird
    rotation_matrix = np.identity(3) + vx + np.dot(vx, vx) * ((1 - c) / s ** 2)
    ## Ende Rechnung
    return rotation_matrix


def calculate_circle_rotation_matrix_2(circle_direction, direction):
    """
    Compute the rotation matrix needed to rotate the vector normal to the circle surface to a specified direction
    """

    # Ab hier Rechnung aus der Dissertation um Rotationsmatrix zu bestimmen
    circle_direction = np.array(circle_direction)
    k = (
            (direction * circle_direction) / (np.linalg.norm(circle_direction))
    )  # Manchmal zeigt der Rotor in die falsche Richtung. Dann muss das Vorzeichen von k angepasst werden
    n = np.array([0, 0, 1])

    v = np.cross(k, n)
    s = np.linalg.norm(v)
    c = np.dot(k, n)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # finale Rotationsmatrix, die num immer zum Ausrichten des Koordinatensystems verwendet wird
    rotation_matrix = np.identity(3) + vx + np.dot(vx, vx) * ((1 - c) / s ** 2)
    ## Ende Rechnung
    return rotation_matrix


def residuals(params, points):
    p0 = params[0:3]
    a = params[3:6]
    a = a / np.linalg.norm(a)
    r = params[6]

    diff = points - p0
    distances = np.linalg.norm(np.cross(diff, a), axis=1)
    return distances - r


def find_blade_axis(hub_points, tip_points):
    # Initial guess via PCA
    mean_pt = np.mean(hub_points, axis=0)

    hub_avg_coord = np.mean(hub_points, axis=0)
    tip_avg_coord = np.mean(tip_points, axis=0)
    init_axis = (tip_avg_coord - hub_avg_coord) / np.linalg.norm(tip_avg_coord - hub_avg_coord)

    init_radius = np.mean(
        np.linalg.norm(np.cross(hub_points - mean_pt, init_axis), axis=1)
    )

    res = least_squares(residuals, [*mean_pt, *init_axis, init_radius], args=(hub_points,))

    p0 = res.x[0:3]
    axis_dir = res.x[3:6] / np.linalg.norm(res.x[3:6])
    radius = abs(res.x[6])

    if np.dot(axis_dir, init_axis) < 0:
        axis_dir = -1 * axis_dir

    # Adjust center point and height along axis bounds
    proj = np.dot(hub_points - p0, axis_dir)
    h_min, h_max = np.min(proj), np.max(proj)

    if np.abs(h_min) > np.abs(h_max):
        h_min, h_max = h_max, h_min
    else:
        pass

    blade_anchor = p0 + ((h_min + h_max) / 2) * axis_dir
    blade_dir = axis_dir
    blade_rad = radius

    return blade_anchor, blade_dir, blade_rad


def find_blade_axis2(hub_points, tip_points):
    # Initial guess via PCA
    mean_pt = np.mean(hub_points, axis=0)

    hub_avg_coord = np.mean(hub_points, axis=0)
    tip_avg_coord = np.mean(tip_points, axis=0)
    init_axis = (tip_avg_coord - hub_avg_coord) / np.linalg.norm(tip_avg_coord - hub_avg_coord)

    init_radius = np.mean(
        np.linalg.norm(np.cross(hub_points - mean_pt, init_axis), axis=1)
    )

    initial_guess = geom3d.Cylinder(anchor_point=mean_pt, radius=init_radius, direction=init_axis)
    fitted_cyl = fit3d.cylinder_fit(hub_points, initial_guess)
    # res = least_squares(residuals, [*mean_pt, *init_axis, init_radius], args=(hub_points,))

    # p0 = res.x[0:3]
    # axis_dir = res.x[3:6] / np.linalg.norm(res.x[3:6])
    # radius = abs(res.x[6])
    #
    # # Adjust center point and height along axis bounds
    # proj = np.dot(hub_points - p0, axis_dir)
    # h_min, h_max = np.min(proj), np.max(proj)
    #
    # if np.abs(h_min) > np.abs(h_max):
    #     h_min, h_max = h_max, h_min
    # else:
    #     pass

    blade_base = fitted_cyl.anchor_point
    blade_axis = fitted_cyl.direction
    blade_radius = fitted_cyl.radius

    return blade_base, blade_axis, blade_radius


def rotation_matrix_z(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def correct_pitch_in_out_file(file_path_queue, pitch_angle, aoi_ids_near_center, blade_name_list,
                              blade_number_of_aoi,
                              corrected_folder):
    # Load .out file as VicDataSet instance, basically a reformed version of read_file function. TODO: I should probably rewrite that one.
    dataset = VicDataSet()
    while True:
        content = file_path_queue.get()
        if content is None:
            break
        out_file_idx, out_file = content

        if dataset.load(out_file) == False:
            print("Could not load data set\n")
            exit(-1)

        size_tot = dataset.matrix_size()
        data = dataset.get_values(["sigma", "X", "Y", "Z", "U", "V", "W", "SIGMA_X", "SIGMA_Y", "SIGMA_Z"])

        visibility = np.where(data["sigma"] < 0, 0, 1)

        coordinates = np.empty((size_tot, 3))
        coordinates[:, 0] = data["X"] + data["U"]
        coordinates[:, 1] = data["Y"] + data["V"]
        coordinates[:, 2] = data["Z"] + data["W"]
        new_coordinates = np.empty_like(coordinates)

        xyz_sigmas = np.empty((size_tot, 3))
        xyz_sigmas[:, 0] = data["SIGMA_X"]
        xyz_sigmas[:, 1] = data["SIGMA_Y"]
        xyz_sigmas[:, 2] = data["SIGMA_Z"]

        aoi_n = dataset.num_data()
        aoi_number_list = []
        index_in_aoi_list = []
        for aoi in range(aoi_n):
            aoi_data = dataset.data(aoi)
            aoi_number_list.extend(aoi_data.matrix_size() * [aoi])
            index_in_aoi_list.extend(np.arange(aoi_data.matrix_size()))
        aoi_number = np.array(aoi_number_list)
        index_in_aoi = np.array(index_in_aoi_list)

        found_idx = np.nonzero(visibility == 1)[0]
        found_coordinates = coordinates[found_idx]
        found_aoi_number = aoi_number[found_idx]

        for blade_idx, blade_name in enumerate(blade_name_list):
            # Find the points in the blade hub
            blade_root_idxs = np.nonzero(found_aoi_number == aoi_ids_near_center[blade_idx])[0]
            blade_root_coords = found_coordinates[blade_root_idxs]

            # Find the AoIs in the current blade
            aoi_in_blade_array = np.nonzero(blade_number_of_aoi == blade_idx)[0]

            for aoi_in_blade in aoi_in_blade_array[1::-1]:
                blade_tip_idxs = np.nonzero(found_aoi_number == aoi_in_blade)[0]

                if blade_tip_idxs.size != 0:
                    blade_tip_coords = found_coordinates[blade_tip_idxs]
                    break
                else:
                    continue

            blade_anchor_pt, blade_dir, blade_rad = find_blade_axis(blade_root_coords, blade_tip_coords)

            # Using the stuff
            blade_point_idxs = np.nonzero(np.isin(found_aoi_number, aoi_in_blade_array))[0]
            blade_point_coords = found_coordinates[blade_point_idxs]

            first_rot = calculate_circle_rotation_matrix_2(blade_dir, 1)
            second_rot = rotation_matrix_z(np.deg2rad(pitch_angle[out_file_idx]))
            third_rot = np.linalg.inv(first_rot)
            total_pitch_rot = np.matmul(third_rot, np.matmul(second_rot, first_rot))

            blade_point_coords_wo_pitch = np.dot(total_pitch_rot,
                                                 (blade_point_coords - blade_anchor_pt).T).T + blade_anchor_pt

            new_coordinates[found_idx[blade_point_idxs]] = blade_point_coords_wo_pitch

        new_u = new_coordinates[:, 0] - data["X"]
        new_v = new_coordinates[:, 1] - data["Y"]
        new_w = new_coordinates[:, 2] - data["Z"]
        dataset.set_values({"U": new_u, "V": new_v, "W": new_w})

        if dataset.save(os.path.join(corrected_folder, os.path.basename(out_file))) == False:
            print("Could not save the dataset\n")
            exit(-1)

        print(
            "\r", "Step 1.5/5: Remove pitch angle from the individual blades... {0} % ".format(
                int(out_file_idx / len(pitch_angle) * 100)), end="")
        # return os.path.join(corrected_folder, os.path.basename(out_file))


if __name__ == "__main__":
    pass
