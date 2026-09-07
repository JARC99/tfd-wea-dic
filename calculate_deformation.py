import numpy as np
import easygui
import pandas as pd
import glob
from matplotlib import pyplot as plt

base = easygui.diropenbox(
    "Select the SchlagSchwenk folder:")

paths = []
paths.append(base)

AOI_name = "Blade_0_AOI_2_"

names = [AOI_name]

diff_list = []
for i in range(len(paths)):

    file_1 = paths[i] + "/" + AOI_name
    full_file_name = glob.glob(file_1 + "*")[0]
    print(full_file_name)
    df = pd.read_csv(full_file_name, sep=";", decimal=',', skiprows=1, header=0)

    sigma = df["sigma [pixel]"].to_numpy()
    deformation_U = df["U [mm]"].to_numpy(copy=True)
    bad_sigmas = np.where(sigma < 0)[0]
    real_U = np.full(len(sigma), 0)
    real_U[bad_sigmas] = 0
    deformation_U[bad_sigmas] = 0
    diff_list.append(deformation_U - real_U)
    plt.plot(deformation_U, label=names[i])

plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Schlagverformung in [mm]")
plt.title("Gemessene Schlagverformung")
plt.show()

if len(diff_list) == 0:
    exit()
else:
    print("Fehler Schlagverformung")
    for i in range(len(diff_list)):
        print(names[i], ": Mean: ", np.mean(np.abs(diff_list[i])), "  Sdt: ", np.std(np.abs(diff_list[i])), "  Max: ",
              np.max(np.abs(diff_list[i])))

for i in range(len(diff_list)):
    plt.plot(diff_list[i], label=names[i] + " diff")
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Fehler Schlagverformung in [mm]")
plt.title("Fehler Schlagverformung")
plt.show()

diff_list = []
for i in range(len(paths)):
    # real_V = df_real[" Schwenk"].to_numpy()
    file_1 = paths[i] + "/" + AOI_name
    full_file_name = glob.glob(file_1 + "*")[0]
    df = pd.read_csv(full_file_name, sep=";", decimal=',', skiprows=1, header=0)
    sigma = df["sigma [pixel]"].to_numpy(copy=True)
    deformation_V = df["V [mm]"].to_numpy(copy=True)
    bad_sigmas = np.where(sigma < 0)[0]
    real_V = np.full(len(sigma), 0)
    real_V[bad_sigmas] = 0
    deformation_V[bad_sigmas] = 0
    real_V[bad_sigmas] = 0
    diff_list.append(deformation_V - real_V)
    plt.plot(deformation_V, label=names[i])

# real_V = df_real[" Schwenk"].to_numpy()
# real_V = np.full(len(sigma), 0)
# plt.plot(real_V, label="real_V")
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Schwenkverformung in [mm]")
plt.title("Gemessene Schwenkverformung")
plt.show()

if len(diff_list) == 0:
    exit()
else:
    print("Fehler Schwenkverformung")
    for i in range(len(diff_list)):
        print(names[i], ": Mean: ", np.mean(np.abs(diff_list[i])), "  Sdt: ", np.std(np.abs(diff_list[i])), "  Max: ",
              np.max(np.abs(diff_list[i])))

for i in range(len(diff_list)):
    plt.plot(diff_list[i], label=names[i] + " diff")
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Fehler Schwenkverformung in [mm]")
plt.title("Fehler Schwenkverformung")
plt.show()
