import numpy as np
import pandas as pd
import glob
import os
from matplotlib import pyplot as plt

path = "G:/8 Facher Vergleich/Original helligkeit nicht angepasst/Altes Verfahren/Torsion"

paths = ["G:/Versuch Rotorblatttorsion/Animierte Anlage/Altes VerfahrenMSKE/Torsion", "G:/Versuch Rotorblatttorsion/Animierte Anlage/Neues VerfahrenMSKE/Torsion", "G:/Versuch Rotorblatttorsion/Animierte Anlage/Altes VerfahrenOSKE/Torsion", "G:/Versuch Rotorblatttorsion/Animierte Anlage/Neues VerfahrenOSKE/Torsion"]
names = ["Altes Verfahren MSKE", "Neues Verfahren MSKE", "Altes Verfahren OSKE", "Neues Verfahren OSKE"]

paths = ["G:/Versuch Rotorblatttorsion/Animierte Anlage/Neues VerfahrenMSKE/Torsion", "G:/Versuch Rotorblatttorsion/Animierte Anlage/Neues VerfahrenOSKE/Torsion", "G:/Versuch Rotorblatttorsion/Animierte Anlage/Test/Torsion"]
names = ["Neues Verfahren MSKE", "Neues Verfahren OSKE", "Test"]

paths = ["G:/Versuch Ausrichtungstest/SchwierigerTS/SchlagSchwenk"]
names = ["Verformung"]




#paths = ["G:/Versuch 3 oder 1 AOI zur El/1AOI/SchlagSchwenk", "G:/Versuch 3 oder 1 AOI zur El/3AOI/SchlagSchwenk", "G:/Versuch 3 oder 1 AOI zur El/Aussen/SchlagSchwenk"]
#names = ["1 AOI", "3 AOI", "Aussen"]


#paths = ["G:/Versuch Rotorblatttorsion/Krummedeich/Altes VerfahrenMSKE/Torsion", "G:/Versuch Rotorblatttorsion/Krummedeich/Altes VerfahrenOSKE/Torsion"]
#names = ["Torsion MSKE", "Torsion OSKE"]

#paths = ["G:/Versuch Rotorblatttorsion/Krummedeich/Altes VerfahrenMSKE/Torsion", "G:/Versuch Rotorblatttorsion/Krummedeich/Altes VerfahrenOSKE/Torsion", "G:/Versuch Rotorblatttorsion/Krummedeich/Neues VerfahrenMSKE/Torsion", "G:/Versuch Rotorblatttorsion/Krummedeich/Neues VerfahrenOSKE/Torsion"]
#names = ["Torsion MSKE", "Torsion OSKE", "Neu Torsion MSKE", "Neu Torsion OSKE"]

#paths = ["G:/Versuch Rotorblatttorsion/Krummedeich/Altes VerfahrenMSKE/SchlagSchwenk", "G:/Versuch Rotorblatttorsion/Krummedeich/Neues VerfahrenMSKE/SchlagSchwenk"]
#names = ["Alt MSKE", "Neu MSKE"]

#paths = ["G:/Blender 3d Punkte Torsionstest/Altes Verfahren/Torsion", "G:/Blender 3d Punkte Torsionstest/Altes Verfahren/Torsion"]
#names = ["Torsion MSKE", "Torsion OSKE"]





paths = ["G:/Versuch Rotorblatttorsion Yaw Anderung/Animation/Yaw Anderung/Alt/MSKE/SchlagSchwenk", 
         "G:/Versuch Rotorblatttorsion Yaw Anderung/Animation/Yaw Anderung/Alt/OSKE/SchlagSchwenk", 
         "G:/Versuch Rotorblatttorsion Yaw Anderung/Animation/ohne Yaw Anderung/Alt/MSKE/SchlagSchwenk",
         "G:/Versuch Rotorblatttorsion Yaw Anderung/Animation/ohne Yaw Anderung/Alt/OSKE/SchlagSchwenk"]
names = ["Yaw Anderung MSKE", "Yaw Anderung OSKE", "ohne Yaw Anderung MSKE", "ohne Yaw Anderung OSKE"]



paths = ["G:/Versuch Rotorblatttorsion Yaw Anderung SchlagSchwenk/Animation/mit yaw Anderung/SchlagSchwenk",
         "G:/Versuch Rotorblatttorsion Yaw Anderung SchlagSchwenk/Animation/ohne yaw Anderung/SchlagSchwenk"]
names = ["Yaw Anderung", "ohne Yaw Anderung"]



paths = ["D:/Masterarbeit Jannis/Versuch Rotorblatttorsion Yaw Anderung SchlagSchwenk/Animation/t1/SchlagSchwenk", "D:/Masterarbeit Jannis/Versuch Rotorblatttorsion Yaw Anderung SchlagSchwenk/Animation/t2/SchlagSchwenk"]
names = ["Durchlauf 1", "Durchlauf 2"]

paths = ["G:/8 Facher Vergleich/Original helligkeit nicht angepasst/SchlagSchwenk"]
names = ["Durchlauf 1"]


file_3 = "G:/Versuch Rotorblatttorsion Yaw Anderung SchlagSchwenk/Direkt exportierte Punkte/BLENDER Blade BLADE003.4.csv"



AOI_name = "Blade_0_AOI_2_"


paths = [r"E:\HiWi Cordova\Messung 19_noyaw\subset\SchlagSchwenk"]
names = ["AAAAAAAAAAA"]

#df_real = pd.read_csv(file_3, sep=";", decimal=',')#[0:201]

#print("--> ", len(df_real))

diff_list = []
for i in range(len(paths)):
#    real_U = -df_real["Schlag"].to_numpy()
    file_1 = paths[i] + "/" + AOI_name
    full_file_name = glob.glob(file_1 + "*")[0]
    print(full_file_name)
    df = pd.read_csv(full_file_name, sep=";", decimal=',', skiprows=1, header=0)
    sigma =  df["sigma [pixel]"].to_numpy()
    deformation_U = df["U [mm]"].to_numpy()
    bad_sigmas = np.where(sigma < 0)[0]
    real_U = np.full(len(sigma), 0)
    real_U[bad_sigmas] = 0
    deformation_U[bad_sigmas] = 0
    real_U[bad_sigmas] = 0
    diff_list.append(deformation_U - real_U)
    plt.plot(deformation_U, label=names[i])

#real_U = -df_real["Schlag"].to_numpy()
#real_U = np.full(len(sigma), 0)
#plt.plot(real_U, label="real_U")
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
        print(names[i],  ": Mean: ", np.mean(np.abs(diff_list[i])),  "  Sdt: ", np.std(np.abs(diff_list[i])),  "  Max: ", np.max(np.abs(diff_list[i])))    

for i in range(len(diff_list)):
    plt.plot(diff_list[i], label=names[i] + " diff")
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Fehler Schlagverformung in [mm]")
plt.title("Fehler Schlagverformung")
plt.show()


diff_list = []
for i in range(len(paths)):
    #real_V = df_real[" Schwenk"].to_numpy()
    file_1 = paths[i] + "/" + AOI_name
    full_file_name = glob.glob(file_1 + "*")[0]
    df = pd.read_csv(full_file_name, sep=";", decimal=',', skiprows=1, header=0)
    sigma =  df["sigma [pixel]"].to_numpy()
    deformation_V = df["V [mm]"].to_numpy()
    bad_sigmas = np.where(sigma < 0)[0]
    real_V = np.full(len(sigma), 0)
    real_V[bad_sigmas] = 0
    deformation_V[bad_sigmas] = 0
    real_V[bad_sigmas] = 0
    diff_list.append(deformation_V - real_V)
    plt.plot(deformation_V, label=names[i])

#real_V = df_real[" Schwenk"].to_numpy()
#real_V = np.full(len(sigma), 0)
#plt.plot(real_V, label="real_V")
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
        print(names[i],  ": Mean: ", np.mean(np.abs(diff_list[i])),  "  Sdt: ", np.std(np.abs(diff_list[i])),  "  Max: ", np.max(np.abs(diff_list[i])))    

for i in range(len(diff_list)):
    plt.plot(diff_list[i], label=names[i] + " diff")
plt.legend()
plt.xlabel("Bildnummer")
plt.ylabel("Fehler Schwenkverformung in [mm]")
plt.title("Fehler Schwenkverformung")
plt.show()

