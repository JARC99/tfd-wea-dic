import glob
import math
import os
# from Blender import *
from multiprocessing import Process, Value, shared_memory, Lock, Condition, Queue, Semaphore

import easygui
import pandas as pd
from VicPy import *  # FIXME: Unlabeled import is not recommended, affetcs code readibility.
from geomfitty import geom3d, fit3d
from matplotlib import pyplot as plt

# Pfad zum Ordner, im welchen sich die Out-Datein befinden, welche direkt nach der Triangulation in Vic-3D generiert wurden. 
# Wenn None, dann wird ein File-Picker Dialog angezeigt (ist zu bevorzugen). 
input_folder = None # TODO: move all input fields together.

# Anzahl an Prozessoren zum Einlesen der Datein
number_of_processes = 8

# Gibt an, wie viele Rotorblaetter ausgewertet werden sollen/können. Dient zur Identifikation der Wurzel-ROI. Ist z.B. nur ein Blatt beklebt, dann eine eins eintregen
number_of_marked_blades = 3

# Liste mit den Variablen, welche in die einzelnen CSV-Datein abgespeichert werden sollen. Der Index ist immer dabei und an ersten Stelle
variables_export_name_out_file = ["X", "Y", "Z", "U", "V", "W", "SIGMA_X", "SIGMA_Y", "SIGMA_Z", "sigma"]

# Liste mit der Benennung der Variablen in der CSV-Datei
variables_export_name_csv_file = ['"Index [1]"', '"X [mm]"', '"Y [mm]"', '"Z [mm]"', '"U [mm]"', '"V [mm]"','"W [mm]"', '"Sigma_X [mm]"', '"Sigma_Y [mm]"', '"Sigma_Z [mm]"', '"sigma [pixel]"']


def put_to_queue(input_out_file_folder, file_path_queue : Queue, number_of_workers):
    """
    Pack die einzelnen Pfad der Out-Dateien in eine Queue, welche dann an mehrere Prozesse geht, welche die Dateien laden.

        input_out_file_folder - Pfad zu dem Ordner, in welchem sich die Out-Dateien befinden
        file_path_queue - Queue, in welche die einzelnen Pfad zu den Out-Datein gepackt werden.
        number_of_workers - Anzahl an Arbeitsprozessen, welche die Out-Dateien verarbeiten. Gibt an wie haeufig am Ende None in die Queue gepackt werden muss
    """
    file_counter = 0
    for file_path  in input_out_file_folder:
        file_path_queue.put((file_counter, file_path),)
        file_counter = file_counter + 1

    for _ in range(number_of_workers):
        file_path_queue.put(None)


def read_file(file_name, subset_index, check_aoi_is_empty = False):
    """
    Liest eine out-Datei ein und speichert eine Teilmenge der Punkte ab.

        file_name - Pfad zu der Out-Datei
        subset_index - Gibt an, welche Subsets zurueckgegeben werden sollen
    """

    data = VicDataSet()
    if (data.load(file_name) == False):
        print("Could not load data set\n\n")
        exit(-1)

    size = 0
    for aoi in range(data.numData()):
        d = data.data(aoi)
        size += d.matrixSize()

    found_array = np.empty((size), int) # gibt an, ob ein Subset gefunden wurde. 1 wenn gefunden, sonst 0
    coordinates = np.empty((size,3), float) # enthaelt die Koordinaten der Subsets
    xyz_sigmas = np.empty((size,3), float) # enthaelt die Sigma-Werte eines Subsets
    aoi_number = np.empty((size), int) # gibt an in welcher AOI ein Subset liegt
    index_in_aoi = np.empty((size), int) # Index des Subsets innerhalb der AOI

    index = 0
    for aoi in range(data.numData()):
        d = data.data(aoi)
        rows = d.asArray(["sigma", "X", "Y", "Z", "U", "V", "W", "SIGMA_X", "SIGMA_Y", "SIGMA_Z"])
        rows = rows.view(np.float32).reshape(rows.shape[0], -1)

        rows = rows.T
        found_array_for_aoi = np.where(rows[0] < 0, 0, 1)

        coordinates_for_aoi = (rows[1:4] + rows[4:7]).T
        xyz_sigma_for_aoi = rows[7:10].T


        if check_aoi_is_empty:
            assert len(np.where(found_array_for_aoi == 1)[0]) > 0, f"In AOI {aoi} sind keine sichtbaren Punkte zu Beginn vorhanden"




        found_array[index : index + len(found_array_for_aoi)] = found_array_for_aoi
        coordinates[index : index + len(coordinates_for_aoi)] = coordinates_for_aoi
        xyz_sigmas[index : index + len(xyz_sigma_for_aoi)] = xyz_sigma_for_aoi
        aoi_number[index : index + len(xyz_sigma_for_aoi)] = np.full((len(xyz_sigma_for_aoi)), aoi)
        index_in_aoi[index : index + len(xyz_sigma_for_aoi)] = np.arange(len(found_array_for_aoi))

        index += len(found_array_for_aoi)

    if subset_index is not None:
        found_array = found_array[subset_index]
        coordinates = coordinates[subset_index]
        xyz_sigmas = xyz_sigmas[subset_index]
        aoi_number = aoi_number[subset_index]
        index_in_aoi = index_in_aoi[subset_index]

    return found_array, coordinates, xyz_sigmas, aoi_number, index_in_aoi



def read_file_mean_aoi_pos_2d(file_name):
    """
    Liest eine out-Datei ein und gibt die 2d Durchschnittskoordinate im Referenzbild jeder AOI zurueck.

        file_name - Pfad zu der Out-Datei
    """
    data = VicDataSet()
    if (data.load(file_name) == False):
        print("Could not load data set\n\n")
        exit(-1)
    coordinates = np.empty((data.numData(),2), float) # enthaelt die Koordinaten der Subsets
    for aoi in range(data.numData()):
        d = data.data(aoi)
        rows = d.asArray(["sigma", "x", "y", "u", "v"])
        rows = rows.view(np.float32).reshape(rows.shape[0], -1)
        rows = rows.T
        found_array_for_aoi = np.where(rows[0] < 0, 0, 1)
        indices_found_temp = np.where(found_array_for_aoi == 1)[0]
        coordinates_for_aoi = (rows[1:3] + rows[3:5]).T
        coordinates[aoi] = np.mean(coordinates_for_aoi[indices_found_temp], axis=0)
    return coordinates




def read_file_pos_2d_at_index(file_name, interesting_subsets_id_vicpy):
    """
    Liest eine out-Datei ein und gibt fuer die vorgegebenen Indexe die Position im Referenzbild zurueck.

        file_name - Pfad zu der Out-Datei
        subset_index - Gibt an, welche Subsets zurueckgegeben werden sollen
    """
    var_ids = []
    data = VicDataSet()
    if (data.load(file_name) == False):
        print("Could not load data set\n\n")
        exit(-1)
    coordinates = np.empty((0,2), int) # enthaelt die Koordinaten der Subsets
    for aoi in range(data.numData()):
        d = data.data(aoi)
        if len(var_ids) == 0:
            for var in ["x", "y", "u", "v"]:
                idx = d.varIndex(var)
                if (idx < 0):
                    print("Could not find variable %s" % var)
                else:
                    var_ids.append(idx)
        values = np.array(d.values(interesting_subsets_id_vicpy[aoi], var_ids))
        values =  np.array([values[0:2] + values[2:4]])
        values = np.round(values)
        coordinates = np.append(coordinates, values, axis=0)

    return coordinates




class SharedMemory:
    """
    Klasse, welche aehnliche Schnittstellen wie eine Queue von multiprocessing hat. Wurde allerdings mit SharedMemory implementiert und ist deshalb schneller.
    Wird benutzt, um den Inhalt der Out-Dateien schnell zum Hauptprozess zu uebermitteln.
    """

    def __init__(self, content):
        """
            content - enhaelt einen Beispiel wie der Inhalt aussieht, der spaeter in die Queue geschrieben wird. Entsprechend content werden die einzelnen Felder in aufgebaut. 
        """
        self.shm_list = []
        self.shape_list = []
        self.dtype_list = []
        self.lock1 = Lock()
        self.shm_ready_to_read = Condition(self.lock1)
        self.global_file_number = Value('i', 0)
        self.lock2 = Lock()
        self.condition = Condition(self.lock2)
        for item in content:
            item_size = int(np.prod(item.shape) * np.dtype(item.dtype).itemsize)
            self.shm_list.append(shared_memory.SharedMemory(create=True, size=item_size))
            self.shape_list.append(item.shape)
            self.dtype_list.append(item.dtype)
        self.shm_ready_to_read.acquire()


    def put(self, file_number, content):
        """
        Pack etwas in den SharedMemory. 
            file_number - Die Nummer der Datei, dessen Inhalt in den SharedMemory gepackt werden soll. Ist dafuer da, dass die Inhalte der Dateien in 
                            der richtigen Reihenfolge in die Queue gepackt werden
            content - Inhalt der Datei.
        """
        with self.condition:
            while True:
                if self.global_file_number.value < file_number:
                    self.condition.wait() # Warte, bis die richtige Dateiennummer erreicht ist.
                else:
                    break
        for i, item in enumerate(content):
            shm_buf = np.ndarray(self.shape_list[i], dtype=self.dtype_list[i], buffer=self.shm_list[i].buf)
            np.copyto(shm_buf, item)
        self.shm_ready_to_read.release() # signalisiert, dass Daten in dem SharedMemory liegen



    def get(self):
        """
        Gibt die Daten zurueck, welche in dem SharedMemory liegen.
        """
        self.shm_ready_to_read.acquire()
        return_list = []
        for i in range(len(self.shm_list)):
            shm_buf = np.ndarray(self.shape_list[i], dtype=self.dtype_list[i], buffer=self.shm_list[i].buf)
            return_list.append(shm_buf[:].copy())
        with self.global_file_number.get_lock():
            self.global_file_number.value += 1
        with self.condition:
            self.condition.notify_all()

        return return_list



def read_file_point_loop(file_name_queue, shared_memory : SharedMemory, subset_index):
    """
    Liest die vorgegebenen Punkte aus den vorgegebenen Dateien und packt sie in den SharedMemory. Dient zum schnellen Einlesen des Punktes, welcher die Kreisbahn beschreibt.

        file_path_queue - Queue, welche die Dateinummer und den Dateinamen der Out-Dateien enthaelt.
        shared_memory - SharedMemory, in welchen die Daten der Out-Dateien gepackt werden sollen.
        subset_index - Index des Dateneintrags, welche abgespeichert werden sollen.
    """

    data = VicDataSet()
    var_names = ["sigma", "X", "Y", "Z", "U", "V", "W"]
    var_ids = []
    while True:
        content = file_name_queue.get()
        if content is None:
            break

        file_number, file_name = content

        if (data.load(file_name) == False):
            print("Could not load data set\n\n")
            exit(-1)

        startindex_roi = 0
        coordinate = None
        for aoi in range(data.numData()):
            d = data.data(aoi)
            current_size = d.matrixSize()
            if subset_index < startindex_roi or subset_index > startindex_roi + current_size:
                startindex_roi += current_size
                continue
            local_index = subset_index - startindex_roi
            startindex_roi += current_size

            if len(var_ids) == 0:
                for var in var_names:
                    idx = d.varIndex(var)
                    if (idx < 0):
                        print("Could not find variable %s" % var)
                    else:
                        var_ids.append(idx)

            values = np.array(d.values(local_index, var_ids))
            coordinate = values[1:4] + values[4:7]

        assert coordinate is not None, "Index Error"

        shared_memory.put(file_number, [values[0] > 0, coordinate])



def read_file_loop(file_name_queue, shared_memory : SharedMemory, subset_index):
    """        
    Ist wie "read_file", allerdings werden mehrere Datein in einem Loop eingelesen und der Inhalt wird in den SharedMemory gepackt.

        file_path_queue - Queue, welche die Dateinummer und den Dateinamen der Out-Dateien enthaelt.
        shared_memory - SharedMemory, in welchen die Daten der Out-Dateien gepackt werden sollen.
        subset_index - Index der Dateieintraege, welche abgespeichert werden sollen.
    """

    data = VicDataSet()

    while True:
        content = file_name_queue.get()
        if content is None:
            break

        file_number, file_name = content

        if (data.load(file_name) == False):
            print("Could not load data set\n\n")
            exit(-1)

        size = 0
        for aoi in range(data.numData()):
            d = data.data(aoi)
            size += d.matrixSize()

        found_array = np.empty((size), int) # gibt an, ob ein Subset gefunden wurde. 1 wenn gefunden, sonst 0
        coordinates = np.empty((size,3), float) # enthaelt die Koordinaten der Subsets
        xyz_sigmas = np.empty((size,3), float) # enthaelt die Sigma-Werte eines Subsets
        aoi_number = np.empty((size), int) # gibt an in welcher AOI ein Subset liegt

        index = 0
        for aoi in range(data.numData()):
            d = data.data(aoi)
            rows = d.asArray(["sigma", "X", "Y", "Z", "U", "V", "W", "SIGMA_X", "SIGMA_Y", "SIGMA_Z"])
            rows = rows.view(np.float32).reshape(rows.shape[0], -1)

            rows = rows.T
            found_array_for_aoi = np.where(rows[0] < 0, 0, 1)

            coordinates_for_aoi = (rows[1:4] + rows[4:7]).T
            xyz_sigma_for_aoi = rows[7:10].T

            found_array[index : index + len(found_array_for_aoi)] = found_array_for_aoi
            coordinates[index : index + len(coordinates_for_aoi)] = coordinates_for_aoi
            xyz_sigmas[index : index + len(xyz_sigma_for_aoi)] = xyz_sigma_for_aoi
            aoi_number[index : index + len(xyz_sigma_for_aoi)] = np.full((len(xyz_sigma_for_aoi)), aoi)

            index += len(found_array_for_aoi)

        if subset_index is not None:
            found_array = found_array[subset_index]
            coordinates = coordinates[subset_index]
            xyz_sigmas = xyz_sigmas[subset_index]
            aoi_number = aoi_number[subset_index]

        shared_memory.put(file_number, [found_array, coordinates, xyz_sigmas, aoi_number])



def process_out_files(file_path_queue, output_path1, output_path2, translation_vector_arg, rotation_matrix_arg, roi_ids_near_center, found_array_first_frame, coordinates_first_frame, shared_mem : SharedMemory, interesting_subsets_id_vicpy):
    """
    Passt die Daten in den Out-Datein an, sodass das Koordinatensystem ausgerichtet ist und eliminiert die Starrkoerperrotation. Danach werden die Out-Datein jeweils erneut abgespeichert.

        file_path_queue - Queue, welche die Dateinummer und den Dateinamen der Out-Dateien enthaelt.
        output_path1 - Pfad des Ordners, in welchem die Out-Dateien abgespeichert werden, bei denen das Koordinatensystem (Naben-Koordinatensystem) angepasst wurde
        output_path2 - Pfad des Ordners, in welchem die Out-Dateien abgespeichert werden, bei denen die Starrkoerperrotation (Rotor-Koordinatensystem) eliminiert
        translation_vector_arg - Translationsvektor zur Anpassung des Koordinatensystems (Naben-Koordinatensystem)
        rotation_matrix_arg - Rotationsmatrix zur Anpassung des Koordinatensystems (Naben-Koordinatensystem)
        roi_ids_near_center - Ids der AOI, welche sich im Mittelpunkt des Rotors befinden. Dient zur Eliminierung der Starrkoerperrotation
        found_array_first_frame - Gibt an, welche Subsets im ersten Frame im Blattwurzelbereich gefunden worden sind. 
        coordinates_first_frame - Koordinaten der Subsets im Blattwurzelbereich
        semaphore - Wird immer freigegeben, wenn eine Datein verarbeitet wurde. Dient zur Anzeige des Fortschritts (Prozentangabe im Terminal). 
    """
    var_ids = []
    data = VicDataSet()
    while True:
        translation = RigidTransformation()
        rotation_obj = Rotation()
        rotation = RigidTransformation()

        content = file_path_queue.get()
        if content is None:
            #print("Gesamtzeit: ", time.time() - overall_timer, "  Nur Anpassen: ", change_timer)
            break
        file_number, file = content
        _, tail = os.path.split(file)

        translation_vector = (translation_vector_arg[0], translation_vector_arg[1], translation_vector_arg[2])
        rotation_matrix = rotation_matrix_arg.copy()

        translation.setTranslation(translation_vector)
        rotation_obj.setMatrix(rotation_matrix)

        rotation.setRotation(rotation_obj)
        if (data.load(file) == False):
            print("Could not load data set\n\n")
            exit(-1)
        data.transform(translation, False)
        data.transform(rotation, False)
        data.save(output_path1 + tail)

        coordinates_for_ret = np.empty((data.numData(), 3, len(variables_export_name_out_file)), float)
        for aoi in range(data.numData()):
            d = data.data(aoi)
            if len(var_ids) == 0:
                for var in variables_export_name_out_file:
                    idx = d.varIndex(var)
                    if idx < 0:
                        print("Could not find variable %s" % var)
                    else:
                        var_ids.append(idx)
            coordinates_for_ret[aoi, 1] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 1], var_ids))
            coordinates_for_ret[aoi, 2] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 2], var_ids))



        found_array = np.empty((0), int)
        coordinates = np.empty((0,3), float)
        for aoi in roi_ids_near_center:
            d = data.data(aoi)

            rows = d.asArray(["sigma", "X", "Y", "Z", "U", "V", "W"])
            rows = rows.view(np.float32).reshape(rows.shape[0], -1)

            rows = rows.T
            found_array_for_aoi = rows[0]
            """
            found_array_not_zero = np.where(rows[0] >= 0)[0]
            bad_indices = np.where(rows[0] > np.median(rows[0][found_array_not_zero]))[0]
            found_array_for_aoi[bad_indices] = -1
            """
            found_array_for_aoi = np.where(found_array_for_aoi < 0, 0, 1)

            coordinates_for_aoi = (rows[1:4] + rows[4:7]).T

            found_array = np.append(found_array, found_array_for_aoi, axis=0)
            coordinates = np.append(coordinates, coordinates_for_aoi, axis=0)


        sum_found_array = found_array_first_frame + found_array
        indices_in_both_frames = np.where(sum_found_array == 2)[0]

        assert len(indices_in_both_frames) > 5, f"Nicht genug Punkte im Blattwurzelbereich vorhanden, um die Eliminierung der Starrkörperrotation durchzuführen. Datei: {file}"

        coordinates_in_ref_frame = coordinates_first_frame[indices_in_both_frames]
        coordinates_in_current_frame = coordinates[indices_in_both_frames]


        found_rot_mat = find_rotation(coordinates_in_ref_frame, coordinates_in_current_frame)
        root_points_current_image_both = np.dot(found_rot_mat, coordinates_in_current_frame.T).T
        found_translation = find_translation(coordinates_in_ref_frame, root_points_current_image_both)

        translation = RigidTransformation()
        translation_vector = (found_translation[0], found_translation[1], found_translation[2])
        translation.setTranslation(translation_vector)

        rotation_obj = Rotation()
        rotation_obj.setMatrix(found_rot_mat)

        rotation = RigidTransformation()
        rotation.setRotation(rotation_obj)


        if file_number > 0:
            data.transform(rotation, True)
            data.transform(translation, True)

        data.save(output_path2 + tail)


        for aoi in range(data.numData()):
            d = data.data(aoi)
            if len(var_ids) == 0:
                for var in variables_export_name_out_file:
                    idx = d.varIndex(var)
                    if (idx < 0):
                        print("Could not find variable %s" % var)
                    else:
                        var_ids.append(idx)
            #coordinates_for_ret[aoi, 1] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 1], var_ids))
            #coordinates_for_ret[aoi, 2] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 2], var_ids))
            coordinates_for_ret[aoi, 0] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 0], var_ids))

        shared_mem.put(file_number, [coordinates_for_ret])



def quaternion_to_rotation_matrix(Q):
    q0 = Q[0]
    qx = Q[1]
    qy = Q[2]
    qz = Q[3]

    r00 = (q0**2 + qx**2 - qy**2 - qz**2)
    r01 = 2 * (qx * qy - q0 * qz)
    r02 = 2 * (qx * qz + q0 * qy)

    r10 = 2 * (qy * qx + q0 * qz)
    r11 = (q0**2 - qx**2 + qy**2 - qz**2)
    r12 = 2 * (qy * qz - q0 * qx)

    r20 = 2 * (qz * qx - q0 * qy)
    r21 = 2 * (qz * qy + q0 * qx)
    r22 = (q0**2 - qx**2 - qy**2 + qz**2)

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

    M = np.empty((3,3), float)

    for i in range(3):
        for j in range(3):
            M[j, i] = np.dot(right_co[:, i], left_co[:, j].T)

    N = np.array([[M[0,0]+M[1,1]+M[2,2],        M[1,2]-M[2,1],          M[2,0]-M[0,2],            M[0,1]-M[1,0]],
                  [M[1,2]-M[2,1],               M[0,0]-M[1,1]-M[2,2],   M[0,1]+M[1,0],            M[2,0]+M[0,2]],
                  [M[2,0]-M[0,2],               M[0,1]+M[1,0],         -M[0,0]+M[1,1]-M[2,2],     M[1,2]+M[2,1]],
                  [M[0,1]-M[1,0],               M[2,0]+M[0,2],          M[1,2]+M[2,1],           -M[0,0]-M[1,1]+M[2,2]]])
    eigenvalues, eigenvectors = np.linalg.eig(N)
    eigenvectors = eigenvectors.T
    max_index = np.argmax(eigenvalues)
    test = eigenvectors[max_index] / np.linalg.norm(eigenvectors[max_index])
    return quaternion_to_rotation_matrix(test)



def find_scale(right_co, left_co_temp, rot_mat):
    """        
    Berechnet einen Skalierungsvektor, um Punkte aus dem Koordinatensystem left_co_temp in Punkte nach dem Koordinatensystem right_co zu ueberfuehren

        right_co - Punkte im ersten Koordinatensystem (Referenz)
        left_co_temp - Die gleichen Punkte mit den Koordinaten nach dem zweiten Koordinatensystem
        rot_mat - Rotationsmatrix, welche mit der Funktion find_rotation gefunden wurde
    """
    mean_right = np.mean(right_co, axis=0)
    mean_left = np.mean(left_co_temp, axis=0)
    right_co = right_co - mean_right
    left_co_temp = left_co_temp - mean_left
    D = np.sum(right_co.T * np.dot(rot_mat, left_co_temp.T), axis=1)
    S = np.sum(left_co_temp.T * left_co_temp.T, axis=1)
    return D / S



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
    return np.array([[1,   0,                      0],
                    [0,    math.cos(x_rot_angle),  -math.sin(x_rot_angle)],
                    [0,    math.sin(x_rot_angle),  math.cos(x_rot_angle)]])



def calculate_circle_rotation_matrix(circle_direction, direction):
    # Ab hier Rechnung aus der Dissertation um Rotationsmatrix zu bestimmen
    circle_direction = np.array(circle_direction )
    k = (direction * circle_direction) / (np.linalg.norm(circle_direction)) # Manchmal zeigt der Rotor in die falsche Richtung. Dann muss das Vorzeichen von k angepasst werden
    n = np.array([1,0,0])

    v = np.cross(k, n)
    s = np.linalg.norm(v)
    c = np.dot(k, n)

    vx = np.array([[0,      -v[2],  v[1]],
                    [v[2],   0,      -v[0]],
                    [-v[1],  v[0],   0]])

    # finale Rotationsmatrix, die num immer zum Ausrichten des Koordinatensystems verwendet wird
    rotation_matrix = np.identity(3) + vx + np.dot(vx, vx) * ((1-c) / s**2)
    ## Ende Rechnung
    return rotation_matrix


if __name__ == '__main__':


    # Den Benutzer den Eingabeordner waehlen lassen...
    if input_folder is None:
        input_folder = easygui.diropenbox("Select Folder with out-Files")
    print("Inputfolder: ", input_folder)
    input_out_file_folder = glob.glob(input_folder + '/*.out')
    #input_out_file_folder = input_out_file_folder[:200]

    #print("Step 1/5: Search for suitable measurement points...", end="\r")
    print('\r', "Step 1/5: Search for suitable measurement points...", end='')

    # Alle Messpunkte in der ersten Out-Datei einlesen. Von diesen wird dann eine Untermenge ausgewaehlt, welche im weiteren Verlauf verarbeitet wird (z.B. zur Ermittlung, welcher Punkt die Kreisbahn
    # zur Ausrichtung des Koordinatensystems beschreiben koennte)
    sum_found_array = None
    sum_xyz_sigmas = None
    test_frame_number = 200
    frame_test_list = range(0, test_frame_number, int(test_frame_number / 10))
    for frame in frame_test_list:
        #print("FRAME: ", frame)
        found_array, coordinates, xyz_sigmas, aoi_number, index_in_aoi = read_file(input_out_file_folder[frame], None, frame == 0)
        sigma_correction = np.max(np.linalg.norm(xyz_sigmas, axis=1))# + np.mean(np.linalg.norm(xyz_sigmas, axis=1))) / 2
        xyz_sigmas_norm = np.linalg.norm(xyz_sigmas, axis=1) + np.where(found_array == 0, sigma_correction, 0)

        if sum_found_array is None:
            sum_found_array = found_array
            sum_xyz_sigmas = xyz_sigmas_norm
        else:
            sum_found_array += found_array
            sum_xyz_sigmas += xyz_sigmas_norm


    """
    points_per_aoi = len(np.unique(aoi_number)) * 200
    mean_found_points_list =  []
    for aoi_id in np.unique(aoi_number):
        found_indices = np.where(aoi_number == aoi_id)[0]
        mean_found_points_list.append(int(np.sum(sum_found_array[found_indices]) / 5))
    mean_found_points_list = np.array(mean_found_points_list)

    diffs = max(mean_found_points_list - points_per_aoi, 0)

    found_array, coordinates, xyz_sigmas, aoi_number, index_in_aoi = read_file(input_out_file_folder[0], None)
    test_subsets_list = [] # Die Subsets in dieser Liste werden im weiteren Verlauf betrachtet

    points_per_aoi = int(len(np.where(found_array == 1)[0]) / len(np.unique(aoi_number)))


    for aoi_id in np.unique(aoi_number):
        found_indices = np.where(aoi_number == aoi_id)[0]
        print("====> ", aoi_id, np.unique(sum_found_array[found_indices], return_counts=True), int(np.sum(sum_found_array[found_indices]) / 5), np.sum(found_array[found_indices]))

    exit()

    """
    test_subsets_list = [] # Die Subsets in dieser Liste werden im weiteren Verlauf betrachtet
    if False:
        for aoi_id in np.unique(aoi_number):
            found_indices = np.where((sum_found_array >= len(frame_test_list) - 3) & (aoi_number == aoi_id))[0]
            sum_good_xyz_sigmas = sum_xyz_sigmas[found_indices]
            good_found = sum_found_array[found_indices]
            step = min(10, int(len(sum_good_xyz_sigmas) / 40))
            for i in range(0, len(found_indices)-step, step):
                best_point_offset = np.argmin([sum_good_xyz_sigmas[i : i+step]])
                test_subsets_list.append(found_indices[i +  best_point_offset])
                #print("Offset: ", best_point_offset, "   Sigma: ", good_found[i +  best_point_offset], "   SumFound: ", sum_good_xyz_sigmas[i +  best_point_offset])
            print(aoi_id, len(found_indices), len(range(0, len(found_indices)-step, step)))

    else:
        test_subsets_list = [] # Die Subsets in dieser Liste werden im weiteren Verlauf betrachtet
        found_array, coordinates, xyz_sigmas, aoi_number, index_in_aoi = read_file(input_out_file_folder[0], None)
        found_indices = np.where(found_array == 1)[0]
        for i in range(0, len(found_indices), 10):
            test_subsets_list.append(found_indices[i])


    test_subsets_list = np.array(test_subsets_list)



    visible_counter = np.ones(len(test_subsets_list), int) # zaehlt, wie haeufig ein Messpunkt sichtbar war
    sigmas = np.zeros(len(test_subsets_list), float) # Der Betrag der Messunsicherheit der einzelnen Messpunkte wird hier fuer jedes Bild addiert
    distance = np.ones(len(test_subsets_list), float) # Zurueckgelegte Distanz des Messpunktes. Daran wird erkannt, welcher Messpunkt an der Blattspitze ist und welche innen liegen
    last_coordinates = None # Enthaelt die Koordiaten der Messpunkte aus dem vorherigen Frame
    found_last_time = None # Enthaelt die Info welcher Messpunkt in vorherigen Bild sichtbar war und welcher nicht. Nur wenn im vorherigen Frame und im aktuellen Frame der Messpunkt sichtbar ist wird die Distanz berechnet


    shared_mem = SharedMemory([found_array[test_subsets_list], coordinates[test_subsets_list], xyz_sigmas[test_subsets_list], aoi_number[test_subsets_list]])
    file_path_queue = Queue(maxsize=30)

    # Prozess, welcher die Pfade der Out-Dateien in eine Queue packt. Aus dieser Queue wird von den verarbeitenden Prozesses dann gelesen
    put_to_queue_process = Process(target=put_to_queue, args=(input_out_file_folder,file_path_queue, number_of_processes))
    put_to_queue_process.start()


    # Prozesse starten, welche die Out-Dateien einlesen. Diese werden geordnet in den Shared_Memory gepackt
    workers = []
    for _ in range(number_of_processes):
        worker = Process(target=read_file_loop, args=(file_path_queue, shared_mem, test_subsets_list))
        worker.start()
        workers.append(worker)


    # Verarbeiten der Daten in den Out-Datein. Ueber shared_mem.get() werden die Daten aus den Out-Dateien in dem Hauptprozess zur Verfuegung gestellt
    file_counter = 0
    for file_path  in input_out_file_folder:
        found_array, coordinates, xyz_sigmas, aoi_number  = shared_mem.get()

        visible_counter = visible_counter + found_array

        indices_found = np.where(found_array == 1)[0]
        sigmas[indices_found] = sigmas[indices_found] + np.linalg.norm(xyz_sigmas[indices_found], axis=1)

        if last_coordinates is None:
            last_coordinates = coordinates
            found_last_time = found_array
            continue
        else:
            indices_found = np.where((found_array + found_last_time) == 2)[0]
            distance[indices_found] = (distance + np.linalg.norm(last_coordinates - coordinates, axis = 1))[indices_found]

            last_coordinates = coordinates.copy()
            found_last_time = found_array

        file_counter = file_counter + 1
        print('\r', "Step 1/5: Search for suitable measurement points... ", int((file_counter / len(input_out_file_folder)) * 100), "%", end='')


    # Warten, bis die Einleseprozesse sich beendet haben
    put_to_queue_process.join()
    for worker in workers:
        worker.join()

    # Messpunkt ermitteln, welcher zur Ermittlung der Kreisbahn benutzt wird. Der Messpunkt sollte haeufig sichbar sein und eine geringe Messunsicherheit haben
    print('\r', "Step 1/5: Search for suitable measurement points...  100 %")
    index_least_movement = np.argmin((distance / visible_counter) * (sigmas / visible_counter) + np.where(visible_counter >= np.max(visible_counter) * 0.9, 1, np.max(distance) * 1000))
    index_bad_point = np.argmin((distance / visible_counter) * (1/(sigmas / visible_counter)))
    index_most_movement = np.argmin((1000/(distance / visible_counter)) * (sigmas / visible_counter) + np.where(visible_counter >= np.max(visible_counter) * 0.9, 1, np.max(distance) * 1000))
    #index_least_movement = index_bad_point

    interesting_subsets = []
    available_aoi_ids = np.unique(ar=aoi_number, return_counts=False)

    # Ermittle fuer jede AOI einen Messpunkt, welcher zur Berechnung der Verformung der AOI gut genutzt werden kann. Dieser sollte haeufig sichtbar sein und die Messunsicherheit sollte gering sein
    list_max_points = []
    id_counter = 0
    for local_aoi_id in available_aoi_ids:
        local_sub_ids = np.where(aoi_number == local_aoi_id)[0]
        id_of_good_subset = id_counter + np.argmin((sigmas[local_sub_ids] / visible_counter[local_sub_ids]) + np.where(visible_counter[local_sub_ids] >= np.max(visible_counter[local_sub_ids]) * 0.9, 1, np.max(distance[local_sub_ids]) * 1000))
        #id_of_good_subset = id_counter + np.argmin((1/(sigmas[local_sub_ids]) / visible_counter[local_sub_ids]) + np.where(visible_counter[local_sub_ids] >= np.max(visible_counter[local_sub_ids]) * 0.9, 1, np.max(distance[local_sub_ids]) * 1000))
        interesting_subsets.append(id_of_good_subset)
        id_counter += len(local_sub_ids)
        list_max_points.append(np.max(visible_counter[local_sub_ids]))
    interesting_subsets = np.array(interesting_subsets)

    #interesting_subsets = [  25,  162,  287,  505,  573,  708,  816,  895, 1096, 1349, 1444, 1530, 1715, 1832, 1927]

    distance_per_frame = distance / visible_counter
    found_array, coordinates, xyz_sigmas, aoi_number, index_in_aoi = read_file(input_out_file_folder[0], test_subsets_list)
    indices_found = np.where(found_array == 1)[0]

    # Ermittel fuer jede AOI die Durchschnittsposition und Durchschnittsgeschwindigkeit der AOI. Wird benutzt, um zu erkennen, ob eine AOI aussen oder innen liegt im Rotor liegt.
    mean_speed_array = np.empty((len(available_aoi_ids)))
    mean_position_array = np.empty((len(available_aoi_ids),3))
    for aoi_index in range(len(available_aoi_ids)):
        indices_for_aoi = np.where(aoi_number == aoi_index)[0]
        indices_for_aoi = np.intersect1d(indices_for_aoi, indices_found)
        distance_per_frame_aoi = distance_per_frame[indices_for_aoi]
        positions_in_aoi = coordinates[indices_for_aoi]
        mean_speed_aoi = np.mean(distance_per_frame_aoi)
        mean_position_aoi = np.mean(positions_in_aoi, axis=0)
        mean_speed_array[aoi_index] = mean_speed_aoi
        mean_position_array[aoi_index] = mean_position_aoi


    # Die AOI, die die geringste Durchschnittsgeschwindigkeit haben werden zur Eliminierung der Starrkoerperrotation benutzt
    roi_ids_near_center = np.argsort(mean_speed_array)[:number_of_marked_blades]
    #roi_ids_near_center_O = np.argsort(mean_speed_array)[:number_of_marked_blades]
    #roi_ids_near_center = np.array([np.argsort(mean_speed_array)[1]])

    # Speicher alle id der Messpunkte, die zu den inneren AOI gehoeren (zu Visualisierungszwecken). 
    indices_of_inner_subsets = np.array([], dtype=int)
    for aoi_id in roi_ids_near_center:
        indices_of_inner_subsets = np.append(indices_of_inner_subsets, np.where(aoi_number == aoi_id)[0])

    # Erzeuge eine Liste, welche fuer jede AOI die id des Rotorblattes enthaelt. Die id 0 muss nicht dem Rotorblatt A entsprechen. 
    blade_number_of_aoi = np.empty((len(available_aoi_ids)))
    for aoi_id in range(len(available_aoi_ids)):
        blade_number_of_aoi[aoi_id] = np.argmin(np.linalg.norm(mean_position_array[aoi_id] - mean_position_array[roi_ids_near_center], axis=1))
        #blade_number_of_aoi[aoi_id] = np.argmin(np.linalg.norm(mean_position_array[aoi_id] - mean_position_array[roi_ids_near_center_O], axis=1))

    # Erzeuge eine Liste, welche fuer jede AOI die AOI-Nummer innerhalb eines Rotorblattes enthaelt. Die innerste AOI auf dem Rotorblatt bekommt die id 0.
    aoi_to_blade_aoi = np.array([])
    for aoi_id in range(len(available_aoi_ids)):
        blade_id = blade_number_of_aoi[aoi_id]
        aoi_to_blade_aoi = np.append(aoi_to_blade_aoi, np.searchsorted(np.sort(mean_speed_array[np.where(blade_number_of_aoi == blade_id)]), mean_speed_array[aoi_id]))

    # Erzeuge Visualisierung, welche die berechneten Informationen in den Messdaten zeigt.    
    found_and_inner_subsets = np.intersect1d(indices_found, indices_of_inner_subsets)
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(coordinates[indices_found].T[0], coordinates[indices_found].T[1], coordinates[indices_found].T[2], c = 'b', alpha=.5, s = 10)
    ax.scatter(coordinates[index_least_movement].T[0], coordinates[index_least_movement].T[1], coordinates[index_least_movement].T[2], c = 'g', s = 180, label='Punkt für Kreisbahn')
    ax.scatter(coordinates[index_most_movement].T[0], coordinates[index_most_movement].T[1], coordinates[index_most_movement].T[2], c = 'r', s = 80, label='Rotorblattspitze')
    ax.scatter(coordinates[found_and_inner_subsets].T[0], coordinates[found_and_inner_subsets].T[1], coordinates[found_and_inner_subsets].T[2], c = 'y', s = 30, label='Blattwurzelbereiche')
    ax.scatter(coordinates[interesting_subsets].T[0], coordinates[interesting_subsets].T[1], coordinates[interesting_subsets].T[2], c = 'm', s = 80, label='Gute Punkte')
    #ax.scatter(coordinates[index_bad_point].T[0], coordinates[index_bad_point].T[1], coordinates[index_bad_point].T[2], c = 'y', s = 80, label='Schlechter Punkt')
    ax.scatter(mean_position_array.T[0], mean_position_array.T[1], mean_position_array.T[2], c = 'cyan', s = 80, label='AOI Mittelpunkt')
    for aoi_id in range(len(available_aoi_ids)):
        ax.text(mean_position_array[aoi_id,0],mean_position_array[aoi_id,1],mean_position_array[aoi_id,2],  '%s' % (str(round(blade_number_of_aoi[aoi_id])) + " " + str(round(aoi_to_blade_aoi[aoi_id]))), size=10, zorder=1, color='k')
    ax.set_title('Gefundene Punkte')
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.set_zlabel('z-Achse')
    ax = plt.gca()
    plt.legend()
    ax.set_aspect('equal', adjustable='box')
    fig.show()
    plt.pause(1)


    #index_least_movement = 225


    # Die Koordinaten des Ermittelten Punktes zur Beschreibung der Kreisbahn werden nun ueber die ganze Messung hinweg eingelesen
    print('\r', "Step 2/5: Create circle... ", end='')

    # SharedMemory, in welchem die Koordinaten dieses Messpunktes abgespeichert werden, um sie im Hauptprozess zu verarbeiten
    shared_mem = SharedMemory([found_array[[0]], coordinates[[0]]])
    file_path_queue = Queue(maxsize=30)

    # Prozess zum Bereitstellen der Pfade zu den Out-Datein
    put_to_queue_process = Process(target=put_to_queue, args=(input_out_file_folder, file_path_queue, number_of_processes))
    put_to_queue_process.start()

    # Prozesse zum Einlesen der Out-Dateien erzeugen
    workers = []
    for _ in range(number_of_processes):
        worker = Process(target=read_file_point_loop, args=(file_path_queue, shared_mem, test_subsets_list[index_least_movement]))
        worker.start()
        workers.append(worker)

    # Array erzeugen, in welchem die Koordinaten abgelegt werden. Wenn der Messpunkt in einem Frame nicht sichtbar ist wird Nichts abgespeichert
    coordinates = np.zeros((len(input_out_file_folder), 3), float)
    file_counter = 0
    not_found_counter = 0

    # Abspeichern der Koordinaten 
    for file_path  in input_out_file_folder:
        found_array, real_points = shared_mem.get()
        if found_array[0] == 1:
            coordinates[file_counter - not_found_counter] = real_points.copy()
        else:
            not_found_counter += 1
        file_counter = file_counter + 1
        print('\r', "Step 2/5: Create circle... ", int((file_counter / len(input_out_file_folder)) * 100), "%", end='')


    put_to_queue_process.join()
    for worker in workers:
        worker.join()

    print('\r', "Step 2/5: Create circle...  100 % ")
    # Array auf die Stelle kuerzen, bis zu welcher die Messpunkt abgespeichert worden sind
    #coordinates = coordinates[0 : file_counter - not_found_counter]

    coordinates = coordinates[0:400] # TODO: Wieder entfernen

    # Parameter des Kreises berechnen. Hierfuer wird die Bibliothek geomfitty verwendet. Diese muss allerdings minimal angepasst werden. Die direkte Version von GitHub kann Probleme bereiten
    print('\r', "Step 3/5: Calculate circle...", end='')
    initial_guess = geom3d.Circle3D(np.mean(coordinates, axis=0), [1,  0,  0], 7)
    circle = fit3d.circle3D_fit(coordinates, initial_guess=initial_guess)

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(coordinates.T[0], coordinates.T[1], coordinates.T[2])
    ax.set_title('Verwendete Kreisbahn zur Ausrichtung des Koordinatensystems')
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.set_zlabel('z-Achse')
    ax = plt.gca()
    fig.show()
    plt.pause(1)


    circle_center = circle.center
    circle_direction = circle.direction


    # Berechne die Rotationsmatrix, um die Kreisbahn in die yz-Ebene zu rotieren
    rotation_matrix = calculate_circle_rotation_matrix(circle_direction, 1)

    # Richte Kreisbahn um die yz-Ebene aus und lege den Koordinatenursprung in die Mittle der Kreisbahn
    coordinates_temp = np.dot(rotation_matrix, (coordinates - circle_center).T).T

    # Im Folgenden wird ermittelt, in welche Richtung sich der verfolgte Punkt um die x-Achse dreht. Also im Uhrzeigersinn oder gegen den Uhrzeigersinn. 
    # Das ist wichtig, damit die x-Achse "in Windrichtung aufsteigend ist". Um die Richtung zu bestimmen wird das Kreuzprodukt aus dem Geschwindigkeitsvektor und dem zugehörigen Messpunkt berechnet.
    direction_array = np.zeros((len(coordinates_temp)-1), np.int8)
    for i in range(len(coordinates_temp)-1):
        pt = coordinates_temp[i]
        v = pt - coordinates_temp[i-1]
        c = np.cross(pt, v)
    if c[0] < 0:
        direction_array[i] = -1
    else:
        direction_array[i] = 1

    if np.mean(direction_array > 0):
        x_axis_alignment = 1
    else:
        x_axis_alignment = -1

    # Die Rotationsmatrix wird erneut berechnet, dieses Mal wird jedoch die Richtung der x-Achse mit berücksichtigt
    rotation_matrix = calculate_circle_rotation_matrix(circle_direction, x_axis_alignment)


    """
    # Ab hier Rechnung aus der Dissertation um Rotationsmatrix zu bestimmen
    circle_direction = np.array(circle_direction )
    k = (x_axis_alignment * circle_direction) / (np.linalg.norm(circle_direction)) # Manchmal zeigt der Rotor in die falsche Richtung. Dann muss das Vorzeichen von k angepasst werden
    n = np.array([1,0,0])
    
    v = np.cross(k, n)
    s = np.linalg.norm(v)
    c = np.dot(k, n)

    vx = np.array([[0,      -v[2],  v[1]],
                    [v[2],   0,      -v[0]],
                    [-v[1],  v[0],   0]])

    # finale Rotationsmatrix, die num immer zum Ausrichten des Koordinatensystems verwendet wird
    rotation_matrix = np.identity(3) + vx + np.dot(vx, vx) * ((1-c) / s**2)
    ## Ende Rechnung
    """


    # Messpunkte aus dem ersten Frame einlesen, um diese zu visualisieren. Hierdurch kann erkannt werden, ob die Messdaten korrekt ausgerichtet werden oder nicht
    found_array, real_points, xyz_sigmas, aoi_number, _  = read_file(input_out_file_folder[0], test_subsets_list)

    # Berechne Rotationsmatrix um die x-Achse, damit das Rotorblatt nach oben zeigt
    #most_moved_point = np.dot(rotation_matrix, (real_points[test_subsets_list[index_most_movement]] - circle_center).T).T
    most_moved_point = np.dot(rotation_matrix, (real_points[index_most_movement] - circle_center).T).T

    diff = most_moved_point / np.linalg.norm(most_moved_point)
    x0 = diff[2]
    x1 = diff[1]
    if x0 > 0:
        angle = math.degrees(math.asin(x1))
    else:
        angle = 360 - math.degrees(math.asin(x1))
        angle += 180
    x_rot_angle = math.radians(angle)
    rot_x = np.array([[1,   0,                      0],
                    [0,    math.cos(x_rot_angle),  -math.sin(x_rot_angle)],
                    [0,    math.sin(x_rot_angle),  math.cos(x_rot_angle)]])


    rotation_matrix = np.dot(rot_x, rotation_matrix)
    coordinates = np.dot(rotation_matrix, (coordinates - circle_center).T).T
    print('\r', "Step 3/5: Calculate circle...  100 %")


    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(coordinates.T[0], coordinates.T[1], coordinates.T[2])
    ax.set_title('Verwendete Kreisbahn zur Ausrichtung des Koordinatensystems')
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.set_zlabel('z-Achse')
    ax = plt.gca()
    fig.show()
    plt.pause(1)


    print('\r', "Step 4/5: Search points for rotor blade torsion calculation...", end='')

    coordinates = np.dot(rotation_matrix, (real_points - circle_center).T).T
    mean_position_array = np.dot(rotation_matrix, (mean_position_array - circle_center).T).T

    rotation_matrix_for_aoi = np.empty((len(available_aoi_ids), 3,3), float)
    for aoi_id in range(len(available_aoi_ids)):
        index_highest_point_on_blade = np.where((blade_number_of_aoi == blade_number_of_aoi[aoi_id]) & (aoi_to_blade_aoi == np.max(aoi_to_blade_aoi)))[0][0]
        rotation_matrix_for_aoi[aoi_id] = find_x_rotation_matrix(mean_position_array[index_highest_point_on_blade])


    indices_found = np.where(found_array == 1)
    indices_for_aoi_inter_org = np.arange(len(coordinates))

    index_array = np.empty((len(available_aoi_ids), 2), int)
    for aoi_index in range(len(available_aoi_ids)):
        current_best_value = float('inf')
        best_points = None

        indices_for_aoi = np.where(aoi_number == aoi_index)[0]
        indices_for_aoi_inter = np.intersect1d(indices_for_aoi, indices_found)
        local_coordinates_for_aoi = coordinates[indices_for_aoi_inter]
        local_coordinates_for_aoi = np.dot(rotation_matrix_for_aoi[aoi_index], local_coordinates_for_aoi.T).T
        local_sigmas = xyz_sigmas[indices_for_aoi_inter]
        local_visible_counter = visible_counter[indices_for_aoi_inter]
        indices_for_aoi_local  = indices_for_aoi_inter_org[indices_for_aoi_inter]
        norm_result = np.linalg.norm(local_sigmas, axis = 1) + 1
        k = 0
        for i in range(len(local_coordinates_for_aoi)):
            for j in range(i):
                p1 = local_coordinates_for_aoi[i]
                p2 = local_coordinates_for_aoi[j]                                                           #TODO: Multiplikation zu + aendern und irgendwie einbauen
                value =  ((abs(p1[2] - p2[2]) + 100) / (abs(p1[1] - p2[1]) + 1) ) * (norm_result[i] + norm_result[j]) * ((np.max(local_visible_counter) + np.max(local_visible_counter)) / (local_visible_counter[i] + local_visible_counter[j]) )
                #print((abs(p1[2] - p2[2]) + 1) / (abs(p1[1] - p2[1]) + 1) )
                if value < current_best_value:
                    current_best_value = value
                    best_points = np.array([indices_for_aoi_local[i], indices_for_aoi_local[j]])
                    #print((abs(p1[2] - p2[2])) / (abs(p1[1] - p2[1]) + 0.1) , (abs(p1[2] - p2[2])), (abs(p1[1] - p2[1]) + 0.1))
        index_array[aoi_index] =  best_points
        print('\r', "Step 4/5: Search points for rotor blade torsion calculation...", int((aoi_index / len(available_aoi_ids)) * 100), "%", end='')


    print('\r', "Step 4/5: Search points for rotor blade torsion calculation...  100 %")


    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(coordinates[indices_found].T[0], coordinates[indices_found].T[1], coordinates[indices_found].T[2], c = 'b', alpha=.5, s = 10)
    ax.scatter(coordinates[index_array].T[0], coordinates[index_array].T[1], coordinates[index_array].T[2], c = 'r', s = 80, label='Gute Punkte fuer Rotorblatttorsion')
    ax.set_title('Ausgerichtetes Koordinatensystem')
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.set_zlabel('z-Achse')
    ax.set_xlim((-60000,60000))
    plt.legend()
    fig.show()
    plt.pause(1)



    # Nun werden die ganzen Messpunkte in den Out-Dateien angepasst und als Kopie abgespeichert
    _, _, _, _, index_in_aoi = read_file(input_out_file_folder[0], test_subsets_list)
    found_array, coordinates, xyz_sigmas, aoi_number, _ = read_file(input_out_file_folder[0], None)

    indices_of_inner_subsets = np.array([], dtype=int)
    for roi_id in roi_ids_near_center:
        indices_of_inner_subsets = np.append(indices_of_inner_subsets, np.where(aoi_number == roi_id)[0])


    found_array_first_frame = found_array[indices_of_inner_subsets]
    coordinates_first_frame = coordinates[indices_of_inner_subsets]
    coordinates_first_frame = np.dot(rotation_matrix, (coordinates_first_frame - circle_center).T).T

    print('\r', "Step 5/5: Store adjusted measurement points...", end='')

    # Anzeigen des Bildes mit Benennung der AOI 
    two_d_coordinates = read_file_mean_aoi_pos_2d(input_out_file_folder[0])
    fig, ax = plt.subplots()
    ax.scatter(two_d_coordinates.T[0], np.max(two_d_coordinates.T[1]) -  two_d_coordinates.T[1])
    plt.title("Bennenung der gefundenen AOI (zum Bild 0 der Kamera 0)")
    for aoi_id in range(len(two_d_coordinates)):
        annotation = "B: " + str(int(blade_number_of_aoi[aoi_id])) + "  A: " + str(int(aoi_to_blade_aoi[aoi_id]))
        ax.annotate(annotation, (two_d_coordinates.T[0][aoi_id], np.max(two_d_coordinates.T[1]) -  two_d_coordinates.T[1][aoi_id]))
    plt.show()
    fig.savefig(input_folder +'/AOI Benennung.png', dpi=fig.dpi)
    plt.pause(1)

    # Erzeuge Ordner, in welchen die angepassten Out-Dateien abgespeichert werden
    if not os.path.isdir(input_folder + "/koordNachGL/"):
        os.mkdir(input_folder + "/koordNachGL/")
    if not os.path.isdir(input_folder + "/koordNachGL_noRot/"):
        os.mkdir(input_folder + "/koordNachGL_noRot/")
    if not os.path.isdir(input_folder + "/SchlagSchwenk/"):
        os.mkdir(input_folder + "/SchlagSchwenk/")
    if not os.path.isdir(input_folder + "/Torsion/"):
        os.mkdir(input_folder + "/Torsion/")

    output_path1 = input_folder + "/koordNachGL/"
    output_path2 = input_folder + "/koordNachGL_noRot/"
    output_path3 = input_folder + "/SchlagSchwenk/"
    output_path4 = input_folder + "/Torsion/"

    # Waehrend der Umformung der Out-Dateien wird pro AOI zusaetzlich ein Messpunkt seperat in einer CSV-Datei abgespeichert. Diese Messpunkte werden per SharedMemory an den Hauptprozess uebergeben, welcher diese dann abspeichert
    interesting_subsets_id_vicpy = np.empty((len(interesting_subsets), 3), int)
    interesting_subsets_id_vicpy[:,0] = index_in_aoi[interesting_subsets]
    interesting_subsets_id_vicpy[:,1] = index_in_aoi[index_array[:,0]]
    interesting_subsets_id_vicpy[:,2] = index_in_aoi[index_array[:,1]]

    #interesting_subsets_id_vicpy[:,1] = [2200,  389, 2484, 3620, 2798, 1947,  762, 3791, 3370, 2493, 1337, 2167,  918,  372,  319]
    #interesting_subsets_id_vicpy[:,2] = [1131,  372, 1881, 2094, 1359, 1148,  417, 1294, 1138,  972,  551, 2109,  888,  325,  293]


    #print("interesting_subsets_id_vicpy[:,1]: ", interesting_subsets_id_vicpy[:,1])
    #print("interesting_subsets_id_vicpy[:,2]: ", interesting_subsets_id_vicpy[:,2])


    position_of_interesting_points_2d = read_file_pos_2d_at_index(input_out_file_folder[0], interesting_subsets_id_vicpy[:,0])
    shared_mem = SharedMemory([np.empty((len(interesting_subsets_id_vicpy), 3, len(variables_export_name_out_file)), np.float32)])
    file_path_queue = Queue(maxsize=30)

    # Prozess zum Bereitstellen der Pfade zu den Out-Datein
    put_to_queue_process = Process(target=put_to_queue, args=(input_out_file_folder, file_path_queue, number_of_processes))
    put_to_queue_process.start()


    semaphore = Semaphore(0) # wird benutzt um die Fortschrittsanzeige zu aktualisieren

    # Verarbeitungsprozesse starten...
    workers = []
    for _ in range(number_of_processes):
        worker = Process(target=process_out_files, args=(file_path_queue, output_path1, output_path2, -circle_center, rotation_matrix, roi_ids_near_center, found_array_first_frame, coordinates_first_frame, shared_mem, interesting_subsets_id_vicpy))
        worker.start()
        workers.append(worker)

    # Die zusaetzlich berechneten Messpunkte fuer die CSV-Datei werden pro Rotorblatt auf die 12-Uhr-Stellung gedreht. WICHTIG: Die Messunsicherheit wird dabei nicht angepasst und ist somit aktuell nicht korrekt.
    # TODO: Messunsicherheit korrekt anpassen
    good_points_data = np.empty((len(input_out_file_folder), len(available_aoi_ids), len(variables_export_name_out_file) ), float)
    good_points_torsion = np.empty((len(input_out_file_folder), len(available_aoi_ids), 2, len(variables_export_name_out_file) ), float)
    for file_counter  in range(len(input_out_file_folder)):
        data = shared_mem.get()
        data = np.array(data)[0]

        for aoi_id in range(len(available_aoi_ids)):
            data[aoi_id, 0, 0:3] = np.dot(rotation_matrix_for_aoi[aoi_id], data[aoi_id, 0, 0:3].T).T
            data[aoi_id, 0, 3:6] = np.dot(rotation_matrix_for_aoi[aoi_id], data[aoi_id, 0, 3:6].T).T
        #    #data[aoi_id, 0, 6:9] = np.dot(rotation_matrix[aoi_id], data[aoi_id, 0, 6:9].T).T
        good_points_data[file_counter] = data[:,0]
        good_points_torsion[file_counter] = data[:,1:]
        print('\r', "Step 5/5: Store adjusted measurement points... ", int((file_counter / len(input_out_file_folder)) * 100), "%", end='')

    """
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(good_points_data.T[0] + good_points_data.T[3], good_points_data.T[1] + good_points_data.T[4], good_points_data.T[2] + good_points_data.T[5], c = 'g', s = 10)
    #ax.scatter(good_points_torsion[0,:,0,0] + good_points_torsion[0,:,0,3], good_points_torsion[0,:,0,1] + good_points_torsion[0,:,0,4], good_points_torsion[0,:,0,2] + good_points_torsion[0,:,0,5], c = 'r', s = 10)
    #ax.scatter(good_points_torsion[0,:,1,0] + good_points_torsion[0,:,1,3], good_points_torsion[0,:,1,1] + good_points_torsion[0,:,1,4], good_points_torsion[0,:,1,2] + good_points_torsion[0,:,1,5], c = 'b', s = 10)
    ax.set_title('Bildnummer: ' + str(file_counter))
    ax.set_xlim((-60000,60000))
    ax.set_ylim((-60000,60000))
    ax.set_zlim((-60000,60000))
    ax.view_init(0, 180, 0)
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.set_zlabel('z-Achse')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()
    """

    # Warte darauf, dass sich die Verarbeitungsprozesse beenden
    put_to_queue_process.join()
    for worker in workers:
        worker.join()

    # Die zusaetzlichen Punkte fuer die AOI abspeichern
    for aoi_id in range(len(available_aoi_ids)):
        dic, dict1, dict2 = {}, {}, {}
        for var_id in range(1, len(variables_export_name_csv_file)):
            dic[variables_export_name_csv_file[var_id]] = good_points_data[:,aoi_id, var_id-1]
            dict1[variables_export_name_csv_file[var_id]] = good_points_torsion[:,aoi_id, 0, var_id-1]
            dict2[variables_export_name_csv_file[var_id]] = good_points_torsion[:,aoi_id, 1, var_id-1]
        df = pd.DataFrame(dic)
        df_1 = pd.DataFrame(dict1)
        df_2 = pd.DataFrame(dict2)

        csv_filename1 = output_path3 + "/Blade_" + str(int(blade_number_of_aoi[aoi_id])) + "_AOI_" + str(int(aoi_to_blade_aoi[aoi_id])) + "_" + str(position_of_interesting_points_2d[aoi_id]) + ".csv"
        csv_filename2 = output_path4 + "/Blade_" + str(int(blade_number_of_aoi[aoi_id])) + "_AOI_" + str(int(aoi_to_blade_aoi[aoi_id])) +  "_P1.csv"
        csv_filename3 = output_path4 + "/Blade_" + str(int(blade_number_of_aoi[aoi_id])) + "_AOI_" + str(int(aoi_to_blade_aoi[aoi_id])) +  "_P2.csv"
        header_text = '"B' + str(int(blade_number_of_aoi[aoi_id])) + " AOI" + str(int(aoi_to_blade_aoi[aoi_id])) + '"' + ";" * len(variables_export_name_out_file)

        # Datei öffnen und Text schreiben
        for csv_file_name in [csv_filename1, csv_filename2, csv_filename3]:
            with open(csv_file_name, "w") as f:
                f.write(header_text + "\n")
        df.to_csv(csv_filename1, index_label=variables_export_name_csv_file[0], mode="a", index=True, quotechar="'", sep=';', decimal=',')
        df_1.to_csv(csv_filename2, index_label=variables_export_name_csv_file[0], mode="a", index=True, quotechar="'", sep=';', decimal=',')
        df_2.to_csv(csv_filename3, index_label=variables_export_name_csv_file[0], mode="a", index=True, quotechar="'", sep=';', decimal=',')


    print('\r', "Step 5/5: Store adjusted measurement points...  100 %")

    exit()

    # Falls notwendig kann fuer jeden Frame eine Visualisierung der Messdaten abgespeichert werden. 

    print("Step 6/5 (Test): Store Images...", end="\r")
    file_counter = 0
    input_out_file_folder = glob.glob(input_folder + '/koordNachGL_noRot/*.out')
    for file_path  in input_out_file_folder:
        print(file_path)
        found_array, real_points, xyz_sigmas, aoi_number, index_in_aoi  = read_file(file_path, None)
        print("Step 6/5 (Test): Store Images... ", int((file_counter / len(input_out_file_folder)) * 100), "%", end="\r")

        indices_found = np.where(found_array == 1)
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection='3d')
        ax.grid()
        ax.scatter(real_points[indices_found].T[0], real_points[indices_found].T[1], real_points[indices_found].T[2], c = 'g', s = 10)
        ax.set_title('Bildnummer: ' + str(file_counter))
        ax.set_xlim((-60000,60000))
        ax.set_ylim((-60000,60000))
        ax.set_zlim((-60000,60000))
        ax.view_init(0, 180, 0)
        ax.set_xlabel('x-Achse')
        ax.set_ylabel('y-Achse')
        ax.set_zlabel('z-Achse')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        fig.savefig('D:/Scatter/fig' + str(file_counter) + '.png', dpi=fig.dpi)
        plt.close()
        file_counter += 1

    print("")





