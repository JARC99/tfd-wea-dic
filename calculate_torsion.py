import easygui
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

paths = []#[r"E:\HiWi Cordova\Messung 19_noyaw\out\DEFAULT\Torsion", r"E:\HiWi Cordova\Messung 19_noyaw\out\YAW_AVE\Torsion", r"E:\HiWi Cordova\Messung 19_noyaw\out\YAW_F0\Torsion"] #[r"E:\HiWi Cordova\Messung 19_noyaw\out\DEFAULT\Torsion", r"E:\HiWi Cordova\Messung 19_noyaw\out\YAW_F0\Torsion"]
paths.append(easygui.diropenbox("Select Folder with out-Files"))#[r"E:\HiWi Cordova\Messung 19_noyaw\out\ANGEL\Torsion"]#[r"E:\HiWi Cordova\Messung24\plots\DEFAULT\Torsion", r"E:\HiWi Cordova\Messung24\plots\YAW_AVE\Torsion", r"E:\HiWi Cordova\Messung24\plots\YAW_F0\Torsion"]
names = ["Angel"]#["Default (Ohne Starrkörper Eliminierung)", "Yaw Correction (Averager Circle)", "Yaw Correction (Frame 0 Circle)"] #["No Yaw Correction","Yaw Correction (Frame 0 Circle)"]

assert len(paths) > 0


# file_3 = "G:/Animierte Anlage/BLENDER Blade BLADE_A_2.csv"
# df_real = pd.read_csv(file_3, sep=";", decimal=',')
# real_angle = -df_real[" Winkel"].to_numpy()
# real_angle = real_angle - real_angle[0]

# Wenn echte Messung, sodass der echte eingestellte Winkel nicht bekannt ist:
real_angle = None


diff_list = []
point_pair_diff_list = []
angles = []

# print(len(real_angle))



for i in range(len(paths)):
    file_1 = paths[i] + "/Blade_0_AOI_0_P1.csv"
    file_2 = paths[i] + "/Blade_0_AOI_0_P2.csv"

    df1 = pd.read_csv(file_1, sep=";", decimal=",", skiprows=1, header=0)
    df2 = pd.read_csv(file_2, sep=";", decimal=",", skiprows=1, header=0)

    sigma_1 = df1["sigma [pixel]"].to_numpy()
    sigma_2 = df2["sigma [pixel]"].to_numpy()
    print(file_1)
    print(len(df1))

    offset1 = np.array(
        [df1["X [mm]"].to_numpy(), df1["Y [mm]"].to_numpy(), df1["Z [mm]"].to_numpy()],
        dtype=float,
    ).T
    offset2 = np.array(
        [df2["X [mm]"].to_numpy(), df2["Y [mm]"].to_numpy(), df2["Z [mm]"].to_numpy()],
        dtype=float,
    ).T

    change1 = np.array(
        [df1["U [mm]"].to_numpy(), df1["V [mm]"].to_numpy(), df1["W [mm]"].to_numpy()]
    ).T
    change2 = np.array(
        [df2["U [mm]"].to_numpy(), df2["V [mm]"].to_numpy(), df2["W [mm]"].to_numpy()]
    ).T

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

    other_diff = np.linalg.norm(diff.T[0:3], axis=0)
    other_diff_relativ = ((other_diff - other_diff[0]) * 1000) / other_diff[0]
    other_diff_relativ[bad_sigmas] = 0
    other_diff_relativ[bad_sigmas] = np.max(other_diff_relativ)
    point_pair_diff_list.append(other_diff_relativ)
    angle[bad_sigmas] = np.nan
    angles.append(angle)
    plt.plot(angle, label=names[i], alpha=0.5)


if len(angle) == len(real_angle):
    plt.plot(real_angle, label="real_angle")
plt.legend()
plt.xlim(0)
plt.xlabel("Bildnummer")
plt.ylabel("Torsionswinkeländerung in °")
plt.title("Gemessenere Torsionswinkeländerung zum Referenzbild")
plt.show()


if len(diff_list) > 0:
    for i in range(len(diff_list)):
        print(
            names[i],
            ": Mean: ",
            np.nanmean(np.abs(diff_list[i])),
            "  Sdt: ",
            np.nanstd(np.abs(diff_list[i])),
            "  Max: ",
            np.nanmax(np.abs(diff_list[i])),
        )

    for i in range(len(diff_list)):
        plt.plot(diff_list[i], label=str(names[i]) + " error")
    plt.legend()
    plt.xlabel("Bildnummer")
    plt.ylabel("Fehler Torsionswinkeländerung in °")
    plt.title("Fehler Torsionswinkeländerung")
    plt.show()


for i in range(len(point_pair_diff_list)):
    plt.plot(point_pair_diff_list[i], label=names[int(i / 1)])
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Änderung Distanz Punkte in mm")
plt.title("Abstand Punktepaare zu Referenzbild")
plt.show()
