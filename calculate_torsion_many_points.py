import glob

import easygui
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

base = easygui.diropenbox(
    "Select Folder with out-Files")
paths = glob.glob(base + "/Tor_*")
assert len(paths) > 0

names = [i for i in range(len(paths))]

# file_3 = "G:/Versuch Rotorblatttorsion Yaw Anderung Torsion/Direkt exportierte Punkte/BLENDER Blade BLADE002.5.csv"
# df_real = pd.read_csv(file_3, sep=";", decimal=',')
# real_angle = df_real[" Winkel"].to_numpy()
# real_angle = real_angle - real_angle[0]
# Wenn  echte Messung, sodass der echte eingestellte Winkel nicht bekannt ist:
real_angle = None

diff_list = []
point_pair_diff_list = []
angles = []

for i in range(len(paths)):

    file_1 = paths[i] + "/Blade_2_AOI_2_P1.csv"
    file_2 = paths[i] + "/Blade_2_AOI_2_P2.csv"

    df1 = pd.read_csv(file_1, sep=";", decimal=',', skiprows=1, header=0)
    df2 = pd.read_csv(file_2, sep=";", decimal=',', skiprows=1, header=0)

    sigma_1 = df1["sigma [pixel]"].to_numpy()
    sigma_2 = df2["sigma [pixel]"].to_numpy()
    print(file_1)

    offset1 = np.array([df1["X [mm]"].to_numpy(), df1["Y [mm]"].to_numpy(), df1["Z [mm]"].to_numpy()]).T
    offset2 = np.array([df2["X [mm]"].to_numpy(), df2["Y [mm]"].to_numpy(), df2["Z [mm]"].to_numpy()]).T

    change1 = np.array([df1["U [mm]"].to_numpy(), df1["V [mm]"].to_numpy(), df1["W [mm]"].to_numpy()]).T
    change2 = np.array([df2["U [mm]"].to_numpy(), df2["V [mm]"].to_numpy(), df2["W [mm]"].to_numpy()]).T

    points1 = offset1 + change1
    points2 = offset2 + change2

    diff = points1 - points2

    bad_sigmas = np.where((sigma_1 < 0) | (sigma_2 < 0))[0]

    distance = np.linalg.norm(diff, axis=1)

    angle = np.arcsin((diff.T[0] / distance))
    angle = np.rad2deg(angle)
    angle = angle - angle[0]

    if real_angle is None:
        real_angle = np.full(len(sigma_1), 0)

    angle[bad_sigmas] = 0
    if len(angle) == len(real_angle):
        angle_diff = angle - real_angle
        angle_diff[bad_sigmas] = np.nan
        diff_list.append(angle_diff)

    # point_pair_diff_list.append(diff_x)
    other_diff = np.linalg.norm(diff.T[0:3], axis=0)
    other_diff_relativ = ((other_diff - other_diff[0]) * 1000) / other_diff[0]
    other_diff_relativ[bad_sigmas] = 0
    other_diff_relativ[bad_sigmas] = np.max(other_diff_relativ)
    point_pair_diff_list.append(other_diff_relativ)
    angle[bad_sigmas] = np.nan
    angles.append(angle)
    plt.plot(angle, label=names[i])

if len(angle) == len(real_angle):
    plt.plot(real_angle, label="real_angle")
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Torsionswinkeländerung in °")
plt.title("Gemessenere Torsionswinkeländerung zum Referenzbild")
plt.show()

if len(diff_list) > 0:
    for i in range(len(diff_list)):
        print(names[i], ": Mean: ", np.nanmean(np.abs(diff_list[i])), "  Sdt: ", np.std(np.abs(diff_list[i])),
              "  Max: ", np.max(np.abs(diff_list[i])))

    for i in range(len(diff_list)):
        plt.plot(diff_list[i], label=str(names[i]) + " error")
    plt.legend()
    plt.xlabel("Bildnummer")
    plt.ylabel("Fehler Torsionswinkeländerung in °")
    plt.title("Fehler Torsionswinkeländerung")
    plt.show()

for i in range(len(point_pair_diff_list)):
    plt.plot(point_pair_diff_list[i], label=names[int(i / 1)])
    # plt.plot(point_pair_diff_list[i], label=names[i] + " diff")
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Änderung Distanz Punkte in mm")
plt.title("Abstand Punktepaare zu Referenzbild")
plt.show()

point_pair_diff_list = np.array(point_pair_diff_list)

mean_point_pair_diff = np.nanmean(np.abs(point_pair_diff_list), axis=1)

better_half = np.sort(mean_point_pair_diff)[len(mean_point_pair_diff) // 2]
good_values = np.where(mean_point_pair_diff < better_half)[0]

for i in range(len(point_pair_diff_list)):
    if i not in good_values:
        plt.plot(point_pair_diff_list[i], label=names[int(i / 1)], color='gray', alpha=.35)
for i in range(len(point_pair_diff_list)):
    if i in good_values:
        plt.plot(point_pair_diff_list[i], label=names[int(i / 1)])

plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Änderung Distanz Punkte")
plt.title("Abstand Punktepaare zu Referenzbild besseren Hälfte an Punktepaaren")
plt.show()

angles = np.array(angles)
angle_mean = np.nanmean(angles[good_values], axis=0)

plt.plot(angle_mean, label="angle_mean", color='black')
if len(angle) == len(real_angle):
    plt.plot(real_angle, label="real_angle")
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Durchschnittswinkeländerung in °")
plt.title("Durchschnittswinkeländerung der besseren Hälfte zum Referenzbild")
plt.show()

filename = paths[0][0:paths[0].rfind("/")] + "/Torsion/mean_angle.csv"
df = pd.DataFrame({"mean_angle": angle_mean})
df.to_csv(filename, index=True, quotechar="'", sep=';', decimal=',')

if len(diff_list) == 0:
    exit()

plt.plot(real_angle - angle_mean, label="real_angle")
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Fehler Durchschnittswinkeländerung in °")
plt.title("Fehler Durchschnittswinkeländerung der besseren Hälfte zum Referenzbild")
plt.show()
