import os
from multiprocessing import (
    Condition,
    Lock,
    Queue,
    Value,
    shared_memory,
)

import numpy as np
from VicPy import (
    RigidTransformation,
    Rotation,
    VicDataSet,
)

from coordsysalign.transformation_fns import find_rotation, find_translation


# -------------------------------------
# --------------------------------------


def put_to_queue(input_out_file_folder, file_path_queue: Queue, number_of_workers):
    """
    Pack die einzelnen Pfad der Out-Dateien in eine Queue, welche dann an mehrere Prozesse geht, welche die Dateien laden.

        input_out_file_folder - Pfad zu dem Ordner, in welchem sich die Out-Dateien befinden
        file_path_queue - Queue, in welche die einzelnen Pfad zu den Out-Datein gepackt werden.
        number_of_workers - Anzahl an Arbeitsprozessen, welche die Out-Dateien verarbeiten. Gibt an wie haeufig am Ende None in die Queue gepackt werden muss
    """
    file_counter = 0
    for file_path in input_out_file_folder:
        file_path_queue.put(
            (file_counter, file_path),
        )
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

    if not data.load(
            file_name
    ):  # Tries to load a .out file into the VicDataSet() instance, it exits if it fails
        print("Could not load data set\n\n")
        exit(-1)

    size = 0
    for aoi in range(
            data.numData()
    ):  # The numData() method returns the number of areas of interest in the dataset
        d = data.data(
            aoi
        )  # Returns a pointer to VicData objet for the specific AOI in the dataset
        size += d.matrixSize()

    found_array = np.empty(
        (size), int
    )  # gibt an, ob ein Subset gefunden wurde. 1 wenn gefunden, sonst 0
    coordinates = np.empty((size, 3), float)  # enthaelt die Koordinaten der Subsets
    xyz_sigmas = np.empty((size, 3), float)  # enthaelt die Sigma-Werte eines Subsets
    aoi_number = np.empty((size), int)  # gibt an in welcher AOI ein Subset liegt
    index_in_aoi = np.empty((size), int)  # Index des Subsets innerhalb der AOI

    index = 0
    for aoi in range(data.numData()):
        d = data.data(aoi)
        rows = d.asArray(
            ["sigma", "X", "Y", "Z", "U", "V", "W", "SIGMA_X", "SIGMA_Y", "SIGMA_Z"]
        )  # TODO: This array is the same as the hardcoded one on top?
        rows = rows.view(np.float32).reshape(rows.shape[0], -1)

        rows = rows.T  # TODO: I'm pretty sure this transpose is not needed
        found_array_for_aoi = np.where(rows[0] < 0, 0, 1)

        coordinates_for_aoi = (rows[1:4] + rows[4:7]).T
        xyz_sigma_for_aoi = rows[7:10].T

        if check_aoi_is_empty:
            assert len(np.where(found_array_for_aoi == 1)[0]) > 0, (
                f"No visible points were found in AoI {aoi}."
            )

        found_array[index: index + len(found_array_for_aoi)] = found_array_for_aoi
        coordinates[index: index + len(coordinates_for_aoi)] = coordinates_for_aoi
        xyz_sigmas[index: index + len(xyz_sigma_for_aoi)] = xyz_sigma_for_aoi
        aoi_number[index: index + len(xyz_sigma_for_aoi)] = np.full(
            (len(xyz_sigma_for_aoi)), aoi
        )
        index_in_aoi[index: index + len(xyz_sigma_for_aoi)] = np.arange(
            len(found_array_for_aoi)
        )

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
    if data.load(file_name) == False:
        print("Could not load data set\n\n")
        exit(-1)

    coordinates = np.empty(
        (data.numData(), 2), float
    )  # enthaelt die Koordinaten der Subsets
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
    if data.load(file_name) == False:
        print("Could not load data set\n\n")
        exit(-1)
    coordinates = np.empty((0, 2), int)  # enthaelt die Koordinaten der Subsets
    for aoi in range(data.numData()):
        d = data.data(aoi)
        if len(var_ids) == 0:
            for var in ["x", "y", "u", "v"]:
                idx = d.varIndex(var)
                if idx < 0:
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
        self.global_file_number = Value("i", 0)
        self.lock2 = Lock()
        self.condition = Condition(self.lock2)
        for item in content:
            item_size = int(np.prod(item.shape) * np.dtype(item.dtype).itemsize)
            self.shm_list.append(
                shared_memory.SharedMemory(create=True, size=item_size)
            )
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
            shm_buf = np.ndarray(
                self.shape_list[i],
                dtype=self.dtype_list[i],
                buffer=self.shm_list[i].buf,
            )
            np.copyto(shm_buf, item)
        self.shm_ready_to_read.release()  # signalisiert, dass Daten in dem SharedMemory liegen

    def get(self):
        """
        Gibt die Daten zurueck, welche in dem SharedMemory liegen.
        """
        self.shm_ready_to_read.acquire()
        return_list = []
        for i in range(len(self.shm_list)):
            shm_buf = np.ndarray(
                self.shape_list[i],
                dtype=self.dtype_list[i],
                buffer=self.shm_list[i].buf,
            )
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

        if data.load(file_name) == False:
            print("Could not load data set\n\n")
            exit(-1)

        startindex_roi = 0
        coordinate = None
        for aoi in range(data.numData()):
            d = data.data(aoi)
            current_size = d.matrixSize()
            if (
                    subset_index < startindex_roi
                    or subset_index > startindex_roi + current_size
            ):
                startindex_roi += current_size
                continue
            local_index = subset_index - startindex_roi
            startindex_roi += current_size

            if len(var_ids) == 0:
                for var in var_names:
                    idx = d.varIndex(var)
                    if idx < 0:
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

        if data.load(file_name) == False:
            print("Could not load data set\n\n")
            exit(-1)

        size = 0
        for aoi in range(data.numData()):
            d = data.data(aoi)
            size += d.matrixSize()

        found_array = np.empty(
            (size), int
        )  # gibt an, ob ein Subset gefunden wurde. 1 wenn gefunden, sonst 0
        coordinates = np.empty((size, 3), float)  # enthaelt die Koordinaten der Subsets
        xyz_sigmas = np.empty(
            (size, 3), float
        )  # enthaelt die Sigma-Werte eines Subsets
        aoi_number = np.empty((size), int)  # gibt an in welcher AOI ein Subset liegt

        index = 0
        for aoi in range(data.numData()):
            d = data.data(aoi)
            rows = d.asArray(
                ["sigma", "X", "Y", "Z", "U", "V", "W", "SIGMA_X", "SIGMA_Y", "SIGMA_Z"]
            )
            rows = rows.view(np.float32).reshape(rows.shape[0], -1)

            rows = rows.T
            found_array_for_aoi = np.where(rows[0] < 0, 0, 1)

            coordinates_for_aoi = (rows[1:4] + rows[4:7]).T
            xyz_sigma_for_aoi = rows[7:10].T

            found_array[index: index + len(found_array_for_aoi)] = found_array_for_aoi
            coordinates[index: index + len(coordinates_for_aoi)] = coordinates_for_aoi
            xyz_sigmas[index: index + len(xyz_sigma_for_aoi)] = xyz_sigma_for_aoi
            aoi_number[index: index + len(xyz_sigma_for_aoi)] = np.full(
                (len(xyz_sigma_for_aoi)), aoi
            )

            index += len(found_array_for_aoi)

        if subset_index is not None:
            found_array = found_array[subset_index]
            coordinates = coordinates[subset_index]
            xyz_sigmas = xyz_sigmas[subset_index]
            aoi_number = aoi_number[subset_index]

        shared_memory.put(
            file_number, [found_array, coordinates, xyz_sigmas, aoi_number]
        )


# def process_out_files(
#     file_path_queue,
#     output_path1,
#     output_path2,
#     translation_vector_arg,
#     rotation_matrix_arg,
#     roi_ids_near_center,
#     found_array_first_frame,
#     coordinates_first_frame,
#     shared_mem: SharedMemory,
#     interesting_subsets_id_vicpy,
#     variables_export_name_out_file,
#     SAVE_OUTPUT_FLAG
# ):
#     """
#     Passt die Daten in den Out-Datein an, sodass das Koordinatensystem ausgerichtet ist und eliminiert die Starrkoerperrotation. Danach werden die Out-Datein jeweils erneut abgespeichert.
#
#         file_path_queue - Queue, welche die Dateinummer und den Dateinamen der Out-Dateien enthält.
#         output_path1 - Pfad des Ordners, in welchem die Out-Dateien abgespeichert werden, bei denen das Koordinatensystem (Naben-Koordinatensystem) angepasst wurde
#         output_path2 - Pfad des Ordners, in welchem die Out-Dateien abgespeichert werden, bei denen die Starrkoerperrotation (Rotor-Koordinatensystem) eliminiert
#         translation_vector_arg - Translationsvektor zur Anpassung des Koordinatensystems (Naben-Koordinatensystem)
#         rotation_matrix_arg - Rotationsmatrix zur Anpassung des Koordinatensystems (Naben-Koordinatensystem)
#         roi_ids_near_center - Ids der AOI, welche sich im Mittelpunkt des Rotors befinden. Dient zur Eliminierung der Starrkoerperrotation
#         found_array_first_frame - Gibt an, welche Subsets im ersten Frame im Blattwurzelbereich gefunden worden sind.
#         coordinates_first_frame - Koordinaten der Subsets im Blattwurzelbereich
#         semaphore - Wird immer freigegeben, wenn eine Datein verarbeitet wurde. Dient zur Anzeige des Fortschritts (Prozentangabe im Terminal).
#     """
#     var_ids = []
#     data = VicDataSet()
#     while True:
#         translation = RigidTransformation()
#         rotation_obj = Rotation()
#         rotation = RigidTransformation()
#
#         content = file_path_queue.get()
#         if content is None:
#             break
#
#         file_number, file = content
#         _, tail = os.path.split(file)
#
#         translation_vector = (
#             translation_vector_arg[0],
#             translation_vector_arg[1],
#             translation_vector_arg[2],
#         )
#         rotation_matrix = rotation_matrix_arg[file_number].copy()
#
#         translation.setTranslation(translation_vector)
#         rotation_obj.setMatrix(rotation_matrix)
#
#         rotation.setRotation(rotation_obj)
#         if data.load(file) == False:
#             print("Could not load data set\n\n")
#             exit(-1)
#         data.transform(translation, False)
#         data.transform(rotation, False)
#
#         if SAVE_OUTPUT_FLAG:
#             data.save(output_path1 + tail)
#         else:
#             pass
#
#         # --------------------------------------------------------------------------------------------------------------
#         # Part 2
#         # --------------------------------------------------------------------------------------------------------------
#
#         # Torsion point pair P1 n
#         coordinates_for_ret = np.empty(
#             (data.numData(), 3, len(variables_export_name_out_file)), float
#         )
#         for aoi in range(data.numData()):
#             d = data.data(aoi)
#             if len(var_ids) == 0:
#                 for var in variables_export_name_out_file:
#                     idx = d.varIndex(var)
#                     if idx < 0:
#                         print("Could not find variable %s" % var)
#                     else:
#                         var_ids.append(idx)
#             coordinates_for_ret[aoi, 1] = np.array(
#                 d.values(interesting_subsets_id_vicpy[aoi, 1], var_ids)
#             )
#             coordinates_for_ret[aoi, 2] = np.array(
#                 d.values(interesting_subsets_id_vicpy[aoi, 2], var_ids)
#             )
#
#         found_array = np.empty((0), int)
#         coordinates = np.empty((0, 3), float)
#         for aoi in roi_ids_near_center:
#             d = data.data(aoi)
#
#             rows = d.asArray(["sigma", "X", "Y", "Z", "U", "V", "W"])
#             rows = rows.view(np.float32).reshape(rows.shape[0], -1)
#
#             rows = rows.T
#             found_array_for_aoi = rows[0]
#             """
#             found_array_not_zero = np.where(rows[0] >= 0)[0]
#             bad_indices = np.where(rows[0] > np.median(rows[0][found_array_not_zero]))[0]
#             found_array_for_aoi[bad_indices] = -1
#             """
#             found_array_for_aoi = np.where(found_array_for_aoi < 0, 0, 1)
#
#             coordinates_for_aoi = (rows[1:4] + rows[4:7]).T
#
#             found_array = np.append(found_array, found_array_for_aoi, axis=0)
#             coordinates = np.append(coordinates, coordinates_for_aoi, axis=0)
#
#         sum_found_array = found_array_first_frame + found_array
#         indices_in_both_frames = np.where(sum_found_array == 2)[0]
#
#         assert len(indices_in_both_frames) > 5, (
#             f"Nicht genug Punkte im Blattwurzelbereich vorhanden, um die Eliminierung der Starrkörperrotation durchzuführen. Datei: {file}"
#         )
#
#         coordinates_in_ref_frame = coordinates_first_frame[indices_in_both_frames]
#         coordinates_in_current_frame = coordinates[indices_in_both_frames]
#
#         found_rot_mat = find_rotation(
#             coordinates_in_ref_frame, coordinates_in_current_frame
#         )
#         root_points_current_image_both = np.dot(
#             found_rot_mat, coordinates_in_current_frame.T
#         ).T
#         found_translation = find_translation(
#             coordinates_in_ref_frame, root_points_current_image_both
#         )
#
#         translation = RigidTransformation()
#         translation_vector = (
#             found_translation[0],
#             found_translation[1],
#             found_translation[2],
#         )
#         translation.setTranslation(translation_vector)
#
#         rotation_obj = Rotation()
#         rotation_obj.setMatrix(found_rot_mat)
#
#         rotation = RigidTransformation()
#         rotation.setRotation(rotation_obj)
#
#         if file_number > 0:
#             data.transform(rotation, True)
#             data.transform(translation, True)
#
#         if SAVE_OUTPUT_FLAG:
#             data.save(output_path2 + tail)
#         else:
#             pass
#
#         for aoi in range(data.numData()):
#             d = data.data(aoi)
#             if len(var_ids) == 0:
#                 for var in variables_export_name_out_file:
#                     idx = d.varIndex(var)
#                     if idx < 0:
#                         print("Could not find variable %s" % var)
#                     else:
#                         var_ids.append(idx)
#             # coordinates_for_ret[aoi, 1] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 1], var_ids))
#             # coordinates_for_ret[aoi, 2] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 2], var_ids))
#             coordinates_for_ret[aoi, 0] = np.array(
#                 d.values(interesting_subsets_id_vicpy[aoi, 0], var_ids)
#             )
#
#         shared_mem.put(file_number, [coordinates_for_ret])


def process_out_files(
        file_path_queue,
        output_path1,
        output_path2,
        translation_vector_arg,
        rotation_matrix_arg,
        roi_ids_near_center,
        found_array_first_frame,
        coordinates_first_frame,
        shared_mem: SharedMemory,
        interesting_subsets_id_vicpy,
        variables_export_name_out_file,
        SAVE_OUTPUT_FLAG
):
    """
    Passt die Daten in den Out-Datein an, sodass das Koordinatensystem ausgerichtet ist und eliminiert die Starrkoerperrotation. Danach werden die Out-Datein jeweils erneut abgespeichert.

        file_path_queue - Queue, welche die Dateinummer und den Dateinamen der Out-Dateien enthält.
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
            break

        file_number, file = content
        _, tail = os.path.split(file)

        translation_vector = (
            translation_vector_arg[0],
            translation_vector_arg[1],
            translation_vector_arg[2],
        )
        rotation_matrix = rotation_matrix_arg[file_number].copy()

        translation.setTranslation(translation_vector)
        rotation_obj.setMatrix(rotation_matrix)

        rotation.setRotation(rotation_obj)
        if data.load(file) == False:
            print("Could not load data set\n\n")
            exit(-1)
        data.transform(translation, False)
        data.transform(rotation, False)

        if SAVE_OUTPUT_FLAG:
            data.save(output_path1 + tail)
        else:
            pass

        # --------------------------------------------------------------------------------------------------------------
        # Part 2
        # --------------------------------------------------------------------------------------------------------------

        # Torsion point pair P1 n
        coordinates_for_ret = np.empty(
            (data.numData(), 3, len(variables_export_name_out_file)), float
        )
        for aoi in range(data.numData()):
            d = data.data(aoi)
            if len(var_ids) == 0:
                for var in variables_export_name_out_file:
                    idx = d.varIndex(var)
                    if idx < 0:
                        print("Could not find variable %s" % var)
                    else:
                        var_ids.append(idx)
            coordinates_for_ret[aoi, 1] = np.array(
                d.values(interesting_subsets_id_vicpy[aoi, 1], var_ids)
            )
            coordinates_for_ret[aoi, 2] = np.array(
                d.values(interesting_subsets_id_vicpy[aoi, 2], var_ids)
            )

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

        assert len(indices_in_both_frames) > 5, (
            f"Nicht genug Punkte im Blattwurzelbereich vorhanden, um die Eliminierung der Starrkörperrotation durchzuführen. Datei: {file}"
        )

        coordinates_in_ref_frame = coordinates_first_frame[indices_in_both_frames]
        coordinates_in_current_frame = coordinates[indices_in_both_frames]

        found_rot_mat = find_rotation(
            coordinates_in_ref_frame, coordinates_in_current_frame
        )
        root_points_current_image_both = np.dot(
            found_rot_mat, coordinates_in_current_frame.T
        ).T
        found_translation = find_translation(
            coordinates_in_ref_frame, root_points_current_image_both
        )

        translation = RigidTransformation()
        translation_vector = (
            found_translation[0],
            found_translation[1],
            found_translation[2],
        )
        translation.setTranslation(translation_vector)

        rotation_obj = Rotation()
        rotation_obj.setMatrix(found_rot_mat)

        rotation = RigidTransformation()
        rotation.setRotation(rotation_obj)

        if file_number > 0:
            data.transform(rotation, True)
            data.transform(translation, True)

        if SAVE_OUTPUT_FLAG:
            data.save(output_path2 + tail)
        else:
            pass

        for aoi in range(data.numData()):
            d = data.data(aoi)
            if len(var_ids) == 0:
                for var in variables_export_name_out_file:
                    idx = d.varIndex(var)
                    if idx < 0:
                        print("Could not find variable %s" % var)
                    else:
                        var_ids.append(idx)
            # coordinates_for_ret[aoi, 1] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 1], var_ids))
            # coordinates_for_ret[aoi, 2] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 2], var_ids))
            coordinates_for_ret[aoi, 0] = np.array(
                d.values(interesting_subsets_id_vicpy[aoi, 0], var_ids)
            )

        shared_mem.put(file_number, [coordinates_for_ret])


def process_out_files_mult_point_tor(file_path_queue, output_path1, output_path2, translation_vector_arg,
                                     rotation_matrix_arg, roi_ids_near_center, found_array_first_frame,
                                     coordinates_first_frame, shared_mem: SharedMemory, interesting_subsets_id_vicpy,
                                     variables_export_name_out_file,
                                     SAVE_OUTPUT_FLAG):
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
            # print("Gesamtzeit: ", time.time() - overall_timer, "  Nur Anpassen: ", change_timer)
            break
        file_number, file = content
        _, tail = os.path.split(file)

        translation_vector = (translation_vector_arg[0], translation_vector_arg[1], translation_vector_arg[2])
        rotation_matrix = rotation_matrix_arg[file_number].copy()

        translation.setTranslation(translation_vector)
        rotation_obj.setMatrix(rotation_matrix)

        rotation.setRotation(rotation_obj)
        if data.load(file) == False:
            print("Could not load data set\n\n")
            exit(-1)
        data.transform(translation, False)
        data.transform(rotation, False)

        if SAVE_OUTPUT_FLAG:
            data.save(output_path1 + tail)
        else:
            pass

        # ------
        # Part 2
        # ------

        coordinates_for_ret = np.empty((data.numData(), 51, len(variables_export_name_out_file)), float)
        for aoi in range(data.numData()):
            d = data.data(aoi)
            if len(var_ids) == 0:
                for var in variables_export_name_out_file:
                    idx = d.varIndex(var)
                    if idx < 0:
                        print("Could not find variable %s" % var)
                    else:
                        var_ids.append(idx)
            # for i in range(1, 51):
            #    coordinates_for_ret[aoi, i] = np.array(d.values(interesting_subsets_id_vicpy[aoi, i], var_ids))

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

        """ToggleNR. Hier lässt sich die Eliminierung der Rotation einstellen. Wird der erste Code Block aktiviert wird die Eliminierung
        durchgeführt. Ist der zweite Block aktiv kommt es nicht zur Eliminierung der Rotation, bevor die Punktepaare zur späteren Bestimmung
        der Torsion exportiert werden."""
        "#1"

        # if file_number > 0:
        #   data.transform(rotation, True)
        #  data.transform(translation, True)

        if False:  # Starrkörperrotation deaktiviert
            data.transform(rotation, True)
            data.transform(translation, True)

        if SAVE_OUTPUT_FLAG:
            data.save(output_path2 + tail)
        else:
            pass

        for aoi in range(data.numData()):
            d = data.data(aoi)
            if len(var_ids) == 0:
                for var in variables_export_name_out_file:
                    idx = d.varIndex(var)
                    if idx < 0:
                        print("Could not find variable %s" % var)
                    else:
                        var_ids.append(idx)
            # coordinates_for_ret[aoi, 1] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 1], var_ids))
            # coordinates_for_ret[aoi, 2] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 2], var_ids))
            coordinates_for_ret[aoi, 0] = np.array(d.values(interesting_subsets_id_vicpy[aoi, 0], var_ids))

            for i in range(1, 51):
                coordinates_for_ret[aoi, i] = np.array(d.values(interesting_subsets_id_vicpy[aoi, i], var_ids))

        shared_mem.put(file_number, [coordinates_for_ret])


if __name__ == "__main__":
    pass
