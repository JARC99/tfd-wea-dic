import numpy as np
import math

def quaternion_to_rotation_matrix(Q):
    q0 = Q[0]
    qx = Q[1]
    qy = Q[2]
    qz = Q[3]

    r00 = (q0 ** 2 + qx ** 2 - qy ** 2 - qz ** 2)
    r01 = 2 * (qx * qy - q0 * qz)
    r02 = 2 * (qx * qz + q0 * qy)

    r10 = 2 * (qy * qx + q0 * qz)
    r11 = (q0 ** 2 - qx ** 2 + qy ** 2 - qz ** 2)
    r12 = 2 * (qy * qz - q0 * qx)

    r20 = 2 * (qz * qx - q0 * qy)
    r21 = 2 * (qz * qy + q0 * qx)
    r22 = (q0 ** 2 - qx ** 2 - qy ** 2 + qz ** 2)

    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
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

    N = np.array([[M[0, 0] + M[1, 1] + M[2, 2], M[1, 2] - M[2, 1], M[2, 0] - M[0, 2], M[0, 1] - M[1, 0]],
                  [M[1, 2] - M[2, 1], M[0, 0] - M[1, 1] - M[2, 2], M[0, 1] + M[1, 0], M[2, 0] + M[0, 2]],
                  [M[2, 0] - M[0, 2], M[0, 1] + M[1, 0], -M[0, 0] + M[1, 1] - M[2, 2], M[1, 2] + M[2, 1]],
                  [M[0, 1] - M[1, 0], M[2, 0] + M[0, 2], M[1, 2] + M[2, 1], -M[0, 0] - M[1, 1] + M[2, 2]]])
    eigenvalues, eigenvectors = np.linalg.eig(N)
    eigenvectors = eigenvectors.T
    max_index = np.argmax(eigenvalues)
    test = eigenvectors[max_index] / np.linalg.norm(eigenvectors[max_index])
    return quaternion_to_rotation_matrix(test)


def find_translation(right_co, left_co_temp):
    """
    Berechnet einen Translationsvektor, um Punkte aus dem Koordinatensystem left_co_temp in Punkte nach dem Koordinatensystem right_co zu ueberfuehren.
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
    return np.array([[1, 0, 0],
                     [0, math.cos(x_rot_angle), -math.sin(x_rot_angle)],
                     [0, math.sin(x_rot_angle), math.cos(x_rot_angle)]])


def calculate_circle_rotation_matrix(circle_direction, direction):
    # Ab hier Rechnung aus der Dissertation um Rotationsmatrix zu bestimmen
    circle_direction = np.array(circle_direction)
    k = (direction * circle_direction) / (np.linalg.norm(
        circle_direction))  # Manchmal zeigt der Rotor in die falsche Richtung. Dann muss das Vorzeichen von k angepasst werden
    n = np.array([1, 0, 0])

    v = np.cross(k, n)
    s = np.linalg.norm(v)
    c = np.dot(k, n)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    # finale Rotationsmatrix, die num immer zum Ausrichten des Koordinatensystems verwendet wird
    rotation_matrix = np.identity(3) + vx + np.dot(vx, vx) * ((1 - c) / s ** 2)
    ## Ende Rechnung
    return rotation_matrix