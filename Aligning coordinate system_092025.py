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
input_folder = None  # TODO: move all input fields together.

# Anzahl an Prozessoren zum Einlesen der Datein
number_of_processes = 16

# Gibt an, wie viele Rotorblaetter ausgewertet werden sollen/können. Dient zur Identifikation der Wurzel-ROI. Ist z.B. nur ein Blatt beklebt, dann eine eins eintregen
number_of_marked_blades = 3

# Liste mit den Variablen, welche in die einzelnen CSV-Datein abgespeichert werden sollen. Der Index ist immer dabei und an ersten Stelle
variables_export_name_out_file = ["X", "Y", "Z", "U", "V", "W", "SIGMA_X", "SIGMA_Y", "SIGMA_Z", "sigma"]

# Liste mit der Benennung der Variablen in der CSV-Datei
variables_export_name_csv_file = ['"Index [1]"', '"X [mm]"', '"Y [mm]"', '"Z [mm]"', '"U [mm]"', '"V [mm]"', '"W [mm]"',
                                  '"Sigma_X [mm]"', '"Sigma_Y [mm]"', '"Sigma_Z [mm]"', '"sigma [pixel]"']


def put_to_queue(input_out_file_folder, file_path_queue: Queue, number_of_workers):
    """
    Pack die einzelnen Pfad der Out-Dateien in eine Queue, welche dann an mehrere Prozesse geht, welche die Dateien laden.

        input_out_file_folder - Pfad zu dem Ordner, in welchem sich die Out-Dateien befinden
        file_path_queue - Queue, in welche die einzelnen Pfad zu den Out-Datein gepackt werden.
        number_of_workers - Anzahl an Arbeitsprozessen, welche die Out-Dateien verarbeiten. Gibt an wie haeufig am Ende None in die Queue gepackt werden muss
    """
    file_counter = 0
    for file_path in input_out_file_folder:
        file_path_queue.put((file_counter, file_path), )
        file_counter = file_counter + 1

    for _ in range(number_of_workers):
        file_path_queue.put(None)


def read_file(file_name, subset_index, check_aoi_is_empty=False):
    """
    Liest eine out-Datei ein und speichert eine Teilmenge der Punkte ab.

        file_name - Pfad zu der Out-Datei
        subset_index - Gibt an, welche Subsets zurueckgegeben werden sollen
    """

    data = VicDataSet()  # Creates an instance of a VicDataSet() class

    if not data.load(file_name):  # Tries to load a .out file into the VicDataSet() instance, it exits if it fails
        print("Could not load data set\n\n")
        exit(-1)

    size = 0
    for aoi in range(data.numData()):  # The numData() method returns the number of areas of interest in the dataset
        d = data.data(aoi)  # Returns a pointer to VicData objet for the specific AOI in the dataset
        size += d.matrixSize()

    found_array = np.empty((size), int)  # gibt an, ob ein Subset gefunden wurde. 1 wenn gefunden, sonst 0
    coordinates = np.empty((size, 3), float)  # enthaelt die Koordinaten der Subsets
    xyz_sigmas = np.empty((size, 3), float)  # enthaelt die Sigma-Werte eines Subsets
    aoi_number = np.empty((size), int)  # gibt an in welcher AOI ein Subset liegt
    index_in_aoi = np.empty((size), int)  # Index des Subsets innerhalb der AOI

    index = 0
    for aoi in range(data.numData()):
        d = data.data(aoi)
        rows = d.asArray(["sigma", "X", "Y", "Z", "U", "V", "W", "SIGMA_X", "SIGMA_Y",
                          "SIGMA_Z"])  #TODO: This array is the same as the hardcoded one on top?
        rows = rows.view(np.float32).reshape(rows.shape[0], -1)

        rows = rows.T  # TODO: I'm pretty sure this transpose is not needed
        found_array_for_aoi = np.where(rows[0] < 0, 0, 1)

        coordinates_for_aoi = (rows[1:4] + rows[4:7]).T
        xyz_sigma_for_aoi = rows[7:10].T

        if check_aoi_is_empty:
            assert len(np.where(found_array_for_aoi == 1)[0]) > 0, f"No visible points were found in AoI {aoi}."

        found_array[index: index + len(found_array_for_aoi)] = found_array_for_aoi
        coordinates[index: index + len(coordinates_for_aoi)] = coordinates_for_aoi
        xyz_sigmas[index: index + len(xyz_sigma_for_aoi)] = xyz_sigma_for_aoi
        aoi_number[index: index + len(xyz_sigma_for_aoi)] = np.full((len(xyz_sigma_for_aoi)),
                                                                    aoi)
        index_in_aoi[index: index + len(xyz_sigma_for_aoi)] = np.arange(len(found_array_for_aoi))

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
    Liest eine out-Datei ein und gibt die 2d Durchschnittskoordinate im Referenzbild jeder AOI zurück.

        file_name - Pfad zu der Out-Datei
    """
    data = VicDataSet()
    if (data.load(file_name) == False):
        print("Could not load data set\n\n")
        exit(-1)

    coordinates = np.empty((data.numData(), 2), float)  # enthaelt die Koordinaten der Subsets
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
    coordinates = np.empty((0, 2), int)  # enthaelt die Koordinaten der Subsets
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
        values = np.array([values[0:2] + values[2:4]])
        values = np.round(values)
        coordinates = np.append(coordinates, values, axis=0)

    return coordinates


class SharedMemory:
    """
    Klasse, welche ähnliche Schnittstellen wie eine Queue von multiprocessing hat. Wurde allerdings mit SharedMemory implementiert und ist deshalb schneller.
    Wird benutzt, um den Inhalt der Out-Dateien schnell zum Hauptprozess zu übermitteln.

    A class with a similar interfaces to those of the Queue class from multiprocessing package. The main difference being the implementation using SharedMemory that makes it faster.
    It is used to feed the contents of the .out files to the main process.
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
                    self.condition.wait()  # Warte, bis die richtige Dateiennummer erreicht ist.
                else:
                    break
        for i, item in enumerate(content):
            shm_buf = np.ndarray(self.shape_list[i], dtype=self.dtype_list[i], buffer=self.shm_list[i].buf)
            np.copyto(shm_buf, item)
        self.shm_ready_to_read.release()  # signalisiert, dass Daten in dem SharedMemory liegen

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


def read_file_point_loop(file_name_queue, shared_memory: SharedMemory, subset_index):
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


def read_file_loop(file_name_queue, shared_memory: SharedMemory, subset_index):
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

        found_array = np.empty((size), int)  # gibt an, ob ein Subset gefunden wurde. 1 wenn gefunden, sonst 0
        coordinates = np.empty((size, 3), float)  # enthaelt die Koordinaten der Subsets
        xyz_sigmas = np.empty((size, 3), float)  # enthaelt die Sigma-Werte eines Subsets
        aoi_number = np.empty((size), int)  # gibt an in welcher AOI ein Subset liegt

        index = 0
        for aoi in range(data.numData()):
            d = data.data(aoi)
            rows = d.asArray(["sigma", "X", "Y", "Z", "U", "V", "W", "SIGMA_X", "SIGMA_Y", "SIGMA_Z"])
            rows = rows.view(np.float32).reshape(rows.shape[0], -1)

            rows = rows.T
            found_array_for_aoi = np.where(rows[0] < 0, 0, 1)

            coordinates_for_aoi = (rows[1:4] + rows[4:7]).T
            xyz_sigma_for_aoi = rows[7:10].T

            found_array[index: index + len(found_array_for_aoi)] = found_array_for_aoi
            coordinates[index: index + len(coordinates_for_aoi)] = coordinates_for_aoi
            xyz_sigmas[index: index + len(xyz_sigma_for_aoi)] = xyz_sigma_for_aoi
            aoi_number[index: index + len(xyz_sigma_for_aoi)] = np.full((len(xyz_sigma_for_aoi)), aoi)

            index += len(found_array_for_aoi)

        if subset_index is not None:
            found_array = found_array[subset_index]
            coordinates = coordinates[subset_index]
            xyz_sigmas = xyz_sigmas[subset_index]
            aoi_number = aoi_number[subset_index]

        shared_memory.put(file_number, [found_array, coordinates, xyz_sigmas, aoi_number])


def process_out_files(file_path_queue, output_path1, output_path2, translation_vector_arg, rotation_matrix_arg,
                      roi_ids_near_center, found_array_first_frame, coordinates_first_frame, shared_mem: SharedMemory,
                      interesting_subsets_id_vicpy):
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
        coordinates = np.empty((0, 3), float)
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

        assert len(
            indices_in_both_frames) > 5, f"Nicht genug Punkte im Blattwurzelbereich vorhanden, um die Eliminierung der Starrkörperrotation durchzuführen. Datei: {file}"

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


if __name__ == '__main__':

    # The script starts off by letting the user choose the folder containing the Vic3D .out files that need to be processed
    if input_folder is None:
        input_folder = easygui.diropenbox("Select Folder with out-Files")
    print("Inputfolder: ", input_folder)
    input_out_file_folder = sorted(glob.glob(input_folder + '/*.out'))  # Sorting files alphabetically

    print('\r', "Step 1/5: Search for suitable measurement points...", end='')

    # We read all the points in the first .out file. A subset of these will be selected and subsequently used for the
    # remaining calculations.
    test_subsets_list = []  # This list will contain the indexes of the chosen points
    found_array, coordinates, xyz_sigmas, aoi_number, index_in_aoi = read_file(input_out_file_folder[0],
                                                                               None)  # Reads the first .out file from the folder. These arrays contain information of all the points in that first frame.
    found_indices = np.where(found_array == 1)[
        0]  # Extracts the indexes of the points tht are visible throughout the file
    for i in range(0, len(found_indices), 10):
        test_subsets_list.append(found_indices[i])  # Fills the index list with 1 out of 10 values of the visible points
    test_subsets_list = np.array(test_subsets_list)

    # Now we prepare to read the remaining of the frames and capture the relevant information from them. The
    # multiprocessing package is used to process these files in parallel and the subset indexes are used so that we
    # do not need to read the whole files. We predefine a set of lists to store the relevant information of each of
    # the observed points throughout the whole recording.
    visible_counter = np.ones(len(test_subsets_list),
                              int)  # Counts how many times a point was visible throughout the frames
    sigmas = np.zeros(len(test_subsets_list),
                      float)  # The magnitude of the uncertainty of the individual points is added for every frame
    distance = np.ones(len(test_subsets_list),
                       float)  # Traveled distance for the point in question throughout the frames.  This is used to determine which point is at the blade tip and which one is in the inner region.
    last_coordinates = None  # Contains the point's coordinates in the previous frame.
    found_last_time = None  # Contains the information of whether a point was visible in the previous image. Only when a point is visible in the current and previous images is the distance calculated

    # Here we initialize the objects needed for the multiprocessing setup. The SharedMemory class was custom written
    # to replace the less efficient default one. The Queu limit is also set here
    shared_mem = SharedMemory(
        [found_array[test_subsets_list], coordinates[test_subsets_list], xyz_sigmas[test_subsets_list],
         aoi_number[test_subsets_list]])  # Initiates an instance of the SharedMemory class
    file_path_queue = Queue(maxsize=30)

    # The process that is being initialized here feeds the .out file paths to the Queue. These files are then taken
    # from this queue and read by the working processes
    put_to_queue_process = Process(target=put_to_queue,
                                   args=(input_out_file_folder, file_path_queue, number_of_processes))
    put_to_queue_process.start()

    # Here we start the processes that will actually read the .out files. Once read, these files are ordered and
    # deposited in the Shared Memory
    workers = []
    for _ in range(number_of_processes):
        worker = Process(target=read_file_loop, args=(file_path_queue, shared_mem, test_subsets_list))
        worker.start()
        workers.append(worker)

    # We extract a representative subset of the data of each an every .out file in the selected directory. This
    # information is initially found in the shared_mem (filled during the multiprocessing process).
    file_counter = 0
    for file_path in input_out_file_folder:
        found_array, coordinates, xyz_sigmas, aoi_number = shared_mem.get()

        visible_counter = visible_counter + found_array # Add to the visible_counter array to see if the point was visible or not in that frame

        indices_found = np.where(found_array == 1)[0] # Determine the indices of the points that are visible for the current frame
        sigmas[indices_found] = sigmas[indices_found] + np.linalg.norm(xyz_sigmas[indices_found], axis=1) # Add the uncertainty values for the visible points

        if last_coordinates is None: # Capture the edge case for the first file
            last_coordinates = coordinates
            found_last_time = found_array
            continue
        else:
            indices_found = np.where((found_array + found_last_time) == 2)[0] # Find the indexes of the points that were visible in the current and the last file
            distance[indices_found] = (distance + np.linalg.norm(last_coordinates - coordinates, axis=1))[indices_found] # Compute the distance traveled between the last and the current frame

            last_coordinates = coordinates.copy() # Prepare the last_coordinates and found_last_time arrays for the next loop of calculations
            found_last_time = found_array

        file_counter = file_counter + 1
        print('\r', "Step 1/5: Search for suitable measurement points... ",
              int((file_counter / len(input_out_file_folder)) * 100), "%", end='')

    # Wait until all the individual reading processes are finished
    put_to_queue_process.join()
    for worker in workers:
        worker.join()

    print('\r', "Step 1/5: Search for suitable measurement points...  100 %")

    # We now need to find a reference point to compute the circular path
    # of the blades. This point will be selected from the subset of studied points. The following considerations must
    # be taken into account:
    # 1. The point must be visible most of the time.
    # 2. The uncertainty of the measurement must be as low as possible.
    index_least_movement = np.argmin((distance / visible_counter) * (sigmas / visible_counter) + np.where(
        visible_counter >= np.max(visible_counter) * 0.9,
        1,
        np.max(distance) * 1000)) # Here we look for the index of the point for which the product of the distance per frame as well and the uncertainty are the smallest. We aditionally set a hard constraint by requiring the point to be visible at least 90% of the time.
    index_most_movement = np.argmin((1000 / (distance / visible_counter)) * (sigmas / visible_counter) + np.where(
        visible_counter >= np.max(visible_counter) * 0.9, 1, np.max(distance) * 1000)) # Here we look for the index of the point for which the product of the inverse distance per frame (multiplied by 1000) and the uncertainty is the smallest. We aditionally set a hard constraint by requiring the point to be visible at least 90% of the time.

    # We now prepare to choose a single point for each AoI that can be later used to measure the deformation of the
    # blade. Similarly to the previous case, this point should be visible most of the time and its uncertainty should
    # be low.
    interesting_subsets = []
    available_aoi_ids = np.unique(ar=aoi_number, return_counts=False) # Determine the number of unique AoIs in the WEA

    #list_max_points = []
    id_counter = 0
    for local_aoi_id in available_aoi_ids:
        local_sub_ids = np.where(aoi_number == local_aoi_id)[0] # Find the indexes of of the points that are located within the current AoI.
        id_of_good_subset = id_counter + np.argmin((sigmas[local_sub_ids] / visible_counter[local_sub_ids]) + np.where(
            visible_counter[local_sub_ids] >= np.max(visible_counter[local_sub_ids]) * 0.9, 1,
            np.max(distance[local_sub_ids]) * 1000)) # Find index of the point within the current AoI which has the smallest uncertainty and is visible at least 90% of the time.
        interesting_subsets.append(id_of_good_subset) # Append that index to the interesting_subsets list.
        id_counter += len(local_sub_ids) # Take into consideration the offset for the computation of the next AoIs.
        #list_max_points.append(np.max(visible_counter[local_sub_ids]))
    interesting_subsets = np.array(interesting_subsets)

    distance_per_frame = distance / visible_counter # Compute the average speed of each of the analyzed points.

    found_array, coordinates, xyz_sigmas, aoi_number, index_in_aoi = read_file(input_out_file_folder[0], # TODO: Re-read the information from the first file (maybe use different variable names for this?).
                                                                               test_subsets_list)
    indices_found = np.where(found_array == 1)[0]

    # Compute the average speed and position of every AoI. This information will be used to determine where on the
    # rotor the AoI is located.
    mean_speed_array = np.empty((len(available_aoi_ids)))
    mean_position_array = np.empty((len(available_aoi_ids), 3))
    for aoi_index in range(len(available_aoi_ids)):
        indices_for_aoi = np.where(aoi_number == aoi_index)[0] # Get the indices of the points located within the current AoI.
        indices_for_aoi = np.intersect1d(indices_for_aoi, indices_found) # Filter out points that are not visible out of the list of indexes.
        distance_per_frame_aoi = distance_per_frame[indices_for_aoi] # Extract the speeds of the relevant points.
        positions_in_aoi = coordinates[indices_for_aoi] # Extract the positions of the relevant points.
        mean_speed_aoi = np.mean(distance_per_frame_aoi) # Compute the average speed of the AoI.
        mean_position_aoi = np.mean(positions_in_aoi, axis=0) # Compute the average position of the AoI.
        mean_speed_array[aoi_index] = mean_speed_aoi # Add the computed mean speed value to the corresponding array for the current AoI.
        mean_position_array[aoi_index] = mean_position_aoi # Add the computed mean position value to the corresponding array for the current AoI.

    # FIXME: The AoIs that have the smallest speed values will be used to eliminate the rigid body rotation. Define which one belongs to blade A.
    roi_ids_near_center = np.argsort(mean_speed_array)[:number_of_marked_blades] # Get the indexes of the (in most cases, 3) smallest AoI speed values.

    idx_bladeA = np.argmax(mean_position_array[roi_ids_near_center, 1])
    idx_bladeB = np.argmin(mean_position_array[roi_ids_near_center, 0])
    idx_bladeC = np.argmax(mean_position_array[roi_ids_near_center, 0])

    roi_ids_near_center = roi_ids_near_center[[idx_bladeA, idx_bladeB, idx_bladeC]]



    # Store all the points that belong to the innermost AoIs (this will be used for visualization).
    indices_of_inner_subsets = np.array([], dtype=int)
    for aoi_id in roi_ids_near_center:
        indices_of_inner_subsets = np.append(indices_of_inner_subsets, np.where(aoi_number == aoi_id)[0])

    # FIXME: Create a list, which contains the rotor blade ID for each of the AoIs. Currently, ID 0 must no necessarily correspond with blade A.
    blade_number_of_aoi = np.empty((len(available_aoi_ids)))
    for aoi_id in range(len(available_aoi_ids)):
        blade_number_of_aoi[aoi_id] = np.argmin(
            np.linalg.norm(mean_position_array[aoi_id] - mean_position_array[roi_ids_near_center], axis=1)) # For each AoI, we compute the distance between its mean location an those of the AoIs closest to the rotor hub. The smallest distance gives away which blade the AoI belongs to.

    # Create a list that contains the AoI number for each of the AoIs within a given rotor blade. The innermost AoI always has the index 0.
    aoi_to_blade_aoi = np.array([])
    for aoi_id in range(len(available_aoi_ids)):
        blade_id = blade_number_of_aoi[aoi_id]
        aoi_to_blade_aoi = np.append(aoi_to_blade_aoi, np.searchsorted(
            np.sort(mean_speed_array[np.where(blade_number_of_aoi == blade_id)]), mean_speed_array[aoi_id]))

    # Plot a diagram with the obtained information.
    blade_name_list = ["A", "B", "C"]
    found_and_inner_subsets = np.intersect1d(indices_found, indices_of_inner_subsets)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(coordinates[indices_found].T[0], coordinates[indices_found].T[1], coordinates[indices_found].T[2], c='b',
               alpha=.5, s=10)
    ax.scatter(coordinates[index_least_movement].T[0], coordinates[index_least_movement].T[1],
               coordinates[index_least_movement].T[2], c='g', s=180, label='Punkt für Kreisbahn')
    ax.scatter(coordinates[index_most_movement].T[0], coordinates[index_most_movement].T[1],
               coordinates[index_most_movement].T[2], c='r', s=80, label='Rotorblattspitze')
    ax.scatter(coordinates[found_and_inner_subsets].T[0], coordinates[found_and_inner_subsets].T[1],
               coordinates[found_and_inner_subsets].T[2], c='y', s=30, label='Blattwurzelbereiche')
    ax.scatter(coordinates[interesting_subsets].T[0], coordinates[interesting_subsets].T[1],
               coordinates[interesting_subsets].T[2], c='m', s=80, label='Gute Punkte')
    ax.scatter(mean_position_array.T[0], mean_position_array.T[1], mean_position_array.T[2], c='cyan', s=80,
               label='AoI Mittelpunkt')
    for aoi_id in range(len(available_aoi_ids)):
        ax.text(mean_position_array[aoi_id, 0], mean_position_array[aoi_id, 1], mean_position_array[aoi_id, 2],
                '%s' % (blade_name_list[int(blade_number_of_aoi[aoi_id])] + ": " + str(round(aoi_to_blade_aoi[aoi_id]))), size=10,
                zorder=1, color='k')
    ax.set_aspect('equal')
    ax.set_title('Gefundene Punkte')
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.set_zlabel('z-Achse')
    ax = plt.gca()
    plt.legend()
    ax.set_aspect('equal', adjustable='box')
    fig.show()
    plt.pause(1)


    # As opposed to the previous case, the coordinates of the point used to define the circular path of the rotor are
    # extracted from the complete dataset.
    print('\r', "Step 2/5: Create circle... ", end='')

    # SharedMemory object were the coordinate of the point are to be stored for later use by the main process and
    # definition of the queue size
    shared_mem = SharedMemory([found_array[[0]], coordinates[[0]]])
    file_path_queue = Queue(maxsize=30)

    # Process that feeds the .out files into the queue
    put_to_queue_process = Process(target=put_to_queue,
                                   args=(input_out_file_folder, file_path_queue, number_of_processes))
    put_to_queue_process.start()

    # Prozesse zum Einlesen der Out-Dateien erzeugen
    workers = []
    for _ in range(number_of_processes):
        worker = Process(target=read_file_point_loop,
                         args=(file_path_queue, shared_mem, test_subsets_list[index_least_movement]))
        worker.start()
        workers.append(worker)

    # Create an array where the point coordinates should be stored. If the chosen point is not visible in a certain frame, nothing is stored.
    coordinates = np.zeros((len(input_out_file_folder), 3), float)
    file_counter = 0
    not_found_counter = 0

    # Store the coordinates for each .out file contianed in the folder.
    for file_path in input_out_file_folder:
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
    # Array auf die Stelle kürzen, bis zu welcher die Messpunkte abgespeichert worden sind # TODO: Check that this slicing is actually appropriate
    coordinates = coordinates[0 : file_counter - not_found_counter]

    # Parameter des Kreises berechnen. Hierfuer wird die Bibliothek geomfitty verwendet. Diese muss allerdings minimal angepasst werden. Die direkte Version von GitHub kann Probleme bereiten
    print('\r', "Step 3/5: Calculate circle...", end='')
    initial_guess = geom3d.Circle3D(np.mean(coordinates, axis=0), [1, 0, 0], 7)
    circle = fit3d.circle3D_fit(coordinates, initial_guess=initial_guess)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(coordinates.T[0], coordinates.T[1], coordinates.T[2])
    ax.set_aspect('equal')
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
    direction_array = np.zeros((len(coordinates_temp) - 1), np.int8)
    for i in range(len(coordinates_temp) - 1):
        pt = coordinates_temp[i]
        v = pt - coordinates_temp[i - 1]
        c = np.cross(pt, v) # TODO: I think that starting from the first and using the last is  mistake
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

    # Messpunkte aus dem ersten Frame einlesen, um diese zu visualisieren. Hierdurch kann erkannt werden, ob die Messdaten korrekt ausgerichtet werden oder nicht
    found_array, real_points, xyz_sigmas, aoi_number, _ = read_file(input_out_file_folder[0], test_subsets_list) # TODO: it's like the fourth time it does this, I'm pretty sure we could just doit once and reuse the values

    # Berechne Rotationsmatrix um die x-Achse, damit das Rotorblatt nach oben zeigt
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
    rot_x = np.array([[1, 0, 0],
                      [0, math.cos(x_rot_angle), -math.sin(x_rot_angle)],
                      [0, math.sin(x_rot_angle), math.cos(x_rot_angle)]])

    rotation_matrix = np.dot(rot_x, rotation_matrix)
    coordinates = np.dot(rotation_matrix, (coordinates - circle_center).T).T
    print('\r', "Step 3/5: Calculate circle...  100 %")

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(coordinates.T[0], coordinates.T[1], coordinates.T[2])
    ax.set_title('Verwendete Kreisbahn zur Ausrichtung des Koordinatensystems')
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.set_zlabel('z-Achse')
    ax = plt.gca()
    ax.set_aspect('equal')
    fig.show()
    plt.pause(1)

    print('\r', "Step 4/5: Search points for rotor blade torsion calculation...", end='')

    coordinates = np.dot(rotation_matrix, (real_points - circle_center).T).T # TODO: is repeated many times to refer to different things throughout the script.
    mean_position_array = np.dot(rotation_matrix, (mean_position_array - circle_center).T).T

    rotation_matrix_for_aoi = np.empty((len(available_aoi_ids), 3, 3), float)
    for aoi_id in range(len(available_aoi_ids)):
        index_highest_point_on_blade = np.where((blade_number_of_aoi == blade_number_of_aoi[aoi_id]) & (aoi_to_blade_aoi == np.max(aoi_to_blade_aoi)))[
            0][0]
        rotation_matrix_for_aoi[aoi_id] = find_x_rotation_matrix(mean_position_array[index_highest_point_on_blade])

    indices_found = np.where(found_array == 1)
    indices_for_aoi_inter_org = np.arange(len(coordinates))

    index_array = np.empty((len(available_aoi_ids), 2), int)
    for aoi_index in range(len(available_aoi_ids)): # TODO: Every time he iterates over the available AoIs, he defines a new iterable with range() or list. A single list could be determine at the beginning and used throughout the script
        current_best_value = float('inf')
        best_points = None

        indices_for_aoi = np.where(aoi_number == aoi_index)[0]
        indices_for_aoi_inter = np.intersect1d(indices_for_aoi, indices_found)
        local_coordinates_for_aoi = coordinates[indices_for_aoi_inter]
        local_coordinates_for_aoi = np.dot(rotation_matrix_for_aoi[aoi_index], local_coordinates_for_aoi.T).T
        local_sigmas = xyz_sigmas[indices_for_aoi_inter]
        local_visible_counter = visible_counter[indices_for_aoi_inter]
        indices_for_aoi_local = indices_for_aoi_inter_org[indices_for_aoi_inter]
        norm_result = np.linalg.norm(local_sigmas, axis=1) + 1
        k = 0
        for i in range(len(local_coordinates_for_aoi)):
            for j in range(i):
                p1 = local_coordinates_for_aoi[i]
                p2 = local_coordinates_for_aoi[j]  #TODO: Multiplikation zu + aendern und irgendwie einbauen
                value = ((abs(p1[2] - p2[2]) + 100) / (abs(p1[1] - p2[1]) + 1)) * (norm_result[i] + norm_result[j]) * (
                            (np.max(local_visible_counter) + np.max(local_visible_counter)) / (
                                local_visible_counter[i] + local_visible_counter[j]))
                #print((abs(p1[2] - p2[2]) + 1) / (abs(p1[1] - p2[1]) + 1) )
                if value < current_best_value:
                    current_best_value = value
                    best_points = np.array([indices_for_aoi_local[i], indices_for_aoi_local[j]])
                    #print((abs(p1[2] - p2[2])) / (abs(p1[1] - p2[1]) + 0.1) , (abs(p1[2] - p2[2])), (abs(p1[1] - p2[1]) + 0.1))
        index_array[aoi_index] = best_points
        print('\r', "Step 4/5: Search points for rotor blade torsion calculation...",
              int((aoi_index / len(available_aoi_ids)) * 100), "%", end='')

    print('\r', "Step 4/5: Search points for rotor blade torsion calculation...  100 %")

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.grid()
    ax.scatter(coordinates[indices_found].T[0], coordinates[indices_found].T[1], coordinates[indices_found].T[2], c='b',
               alpha=.5, s=10)
    ax.scatter(coordinates[index_array].T[0], coordinates[index_array].T[1], coordinates[index_array].T[2], c='r', s=80,
               label='Gute Punkte fuer Rotorblatttorsion')
    ax.set_aspect('equal')
    ax.set_title('Ausgerichtetes Koordinatensystem')
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.set_zlabel('z-Achse')
    ax.set_xlim((-60000, 60000))
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
    ax.scatter(two_d_coordinates.T[0], np.max(two_d_coordinates.T[1]) - two_d_coordinates.T[1])
    plt.title("Bennenung der gefundenen AOI (zum Bild 0 der Kamera 0)")
    for aoi_id in range(len(two_d_coordinates)):
        annotation = "Blade " + blade_name_list[int(blade_number_of_aoi[aoi_id])] + " - AoI " + str(
            int(aoi_to_blade_aoi[aoi_id]))  #TODO: use formated strings
        ax.annotate(annotation,
                    (two_d_coordinates.T[0][aoi_id], np.max(two_d_coordinates.T[1]) - two_d_coordinates.T[1][aoi_id]))
    plt.show()
    fig.savefig(input_folder + '/AOI Benennung.png', dpi=fig.dpi)
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

    # Waehrend der Umformung der Out-Dateien wird pro AoI zusaetzlich ein Messpunkt separat in einer CSV-Datei abgespeichert. Diese Messpunkte werden per SharedMemory an den Hauptprozess uebergeben, welcher diese dann abspeichert
    interesting_subsets_id_vicpy = np.empty((len(interesting_subsets), 3), int)
    interesting_subsets_id_vicpy[:, 0] = index_in_aoi[interesting_subsets]
    interesting_subsets_id_vicpy[:, 1] = index_in_aoi[index_array[:, 0]]
    interesting_subsets_id_vicpy[:, 2] = index_in_aoi[index_array[:, 1]]

    position_of_interesting_points_2d = read_file_pos_2d_at_index(input_out_file_folder[0],
                                                                  interesting_subsets_id_vicpy[:, 0])
    shared_mem = SharedMemory(
        [np.empty((len(interesting_subsets_id_vicpy), 3, len(variables_export_name_out_file)), np.float32)])
    file_path_queue = Queue(maxsize=30)

    # Prozess zum Bereitstellen der Pfade zu den Out-Datein
    put_to_queue_process = Process(target=put_to_queue,
                                   args=(input_out_file_folder, file_path_queue, number_of_processes))
    put_to_queue_process.start()

    semaphore = Semaphore(0)  # wird benutzt um die Fortschrittsanzeige zu aktualisieren

    # Verarbeitungsprozesse starten...
    workers = []
    for _ in range(number_of_processes):
        worker = Process(target=process_out_files,
                         args=(file_path_queue, output_path1, output_path2, -circle_center, rotation_matrix,
                               roi_ids_near_center, found_array_first_frame, coordinates_first_frame, shared_mem,
                               interesting_subsets_id_vicpy))
        worker.start()
        workers.append(worker)

    # Die zusaetzlich berechneten Messpunkte fuer die CSV-Datei werden pro Rotorblatt auf die 12-Uhr-Stellung gedreht. WICHTIG: Die Messunsicherheit wird dabei nicht angepasst und ist somit aktuell nicht korrekt.
    # TODO: Messunsicherheit korrekt anpassen
    good_points_data = np.empty(
        (len(input_out_file_folder), len(available_aoi_ids), len(variables_export_name_out_file)), float)
    good_points_torsion = np.empty(
        (len(input_out_file_folder), len(available_aoi_ids), 2, len(variables_export_name_out_file)), float)
    for file_counter in range(len(input_out_file_folder)):
        data = shared_mem.get()
        data = np.array(data)[0]

        for aoi_id in range(len(available_aoi_ids)):
            data[aoi_id, 0, 0:3] = np.dot(rotation_matrix_for_aoi[aoi_id], data[aoi_id, 0, 0:3].T).T
            data[aoi_id, 0, 3:6] = np.dot(rotation_matrix_for_aoi[aoi_id], data[aoi_id, 0, 3:6].T).T
        #    #data[aoi_id, 0, 6:9] = np.dot(rotation_matrix[aoi_id], data[aoi_id, 0, 6:9].T).T
        good_points_data[file_counter] = data[:, 0]
        good_points_torsion[file_counter] = data[:, 1:]
        print('\r', "Step 5/5: Store adjusted measurement points... ",
              int((file_counter / len(input_out_file_folder)) * 100), "%", end='')

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
            dic[variables_export_name_csv_file[var_id]] = good_points_data[:, aoi_id, var_id - 1]
            dict1[variables_export_name_csv_file[var_id]] = good_points_torsion[:, aoi_id, 0, var_id - 1]
            dict2[variables_export_name_csv_file[var_id]] = good_points_torsion[:, aoi_id, 1, var_id - 1]
        df = pd.DataFrame(dic)
        df_1 = pd.DataFrame(dict1)
        df_2 = pd.DataFrame(dict2)

        csv_filename1 = output_path3 + "/Blade_" + str(int(blade_number_of_aoi[aoi_id])) + "_AOI_" + str(
            int(aoi_to_blade_aoi[aoi_id])) + "_" + str(position_of_interesting_points_2d[aoi_id]) + ".csv"
        csv_filename2 = output_path4 + "/Blade_" + str(int(blade_number_of_aoi[aoi_id])) + "_AOI_" + str(
            int(aoi_to_blade_aoi[aoi_id])) + "_P1.csv"
        csv_filename3 = output_path4 + "/Blade_" + str(int(blade_number_of_aoi[aoi_id])) + "_AOI_" + str(
            int(aoi_to_blade_aoi[aoi_id])) + "_P2.csv"
        header_text = '"B' + str(int(blade_number_of_aoi[aoi_id])) + " AOI" + str(
            int(aoi_to_blade_aoi[aoi_id])) + '"' + ";" * len(variables_export_name_out_file)

        # Datei öffnen und Text schreiben
        for csv_file_name in [csv_filename1, csv_filename2, csv_filename3]:
            with open(csv_file_name, "w") as f:
                f.write(header_text + "\n")
        df.to_csv(csv_filename1, index_label=variables_export_name_csv_file[0], mode="a", index=True, quotechar="'",
                  sep=';', decimal=',')
        df_1.to_csv(csv_filename2, index_label=variables_export_name_csv_file[0], mode="a", index=True, quotechar="'",
                    sep=';', decimal=',')
        df_2.to_csv(csv_filename3, index_label=variables_export_name_csv_file[0], mode="a", index=True, quotechar="'",
                    sep=';', decimal=',')

    print('\r', "Step 5/5: Store adjusted measurement points...  100 %")

    exit()

    # Falls notwendig kann fuer jeden Frame eine Visualisierung der Messdaten abgespeichert werden. 

    print("Step 6/5 (Test): Store Images...", end="\r")
    file_counter = 0
    input_out_file_folder = glob.glob(input_folder + '/koordNachGL_noRot/*.out')
    for file_path in input_out_file_folder:
        print(file_path)
        found_array, real_points, xyz_sigmas, aoi_number, index_in_aoi = read_file(file_path, None)
        print("Step 6/5 (Test): Store Images... ", int((file_counter / len(input_out_file_folder)) * 100), "%",
              end="\r")

        indices_found = np.where(found_array == 1)
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.grid()
        ax.scatter(real_points[indices_found].T[0], real_points[indices_found].T[1], real_points[indices_found].T[2],
                   c='g', s=10)
        ax.set_title('Bildnummer: ' + str(file_counter))
        ax.set_xlim((-60000, 60000))
        ax.set_ylim((-60000, 60000))
        ax.set_zlim((-60000, 60000))
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
