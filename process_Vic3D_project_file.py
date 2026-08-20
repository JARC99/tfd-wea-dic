import numpy as np
import os
import pandas as pd
import math
import xml.etree.ElementTree as ET
from zipfile import ZipFile 
from multiprocessing import Process, Value, shared_memory, Lock, Condition, Queue
import timeit
 

# Ordner, in der die CSV-Datein fuer die Ausrichtung der AOI sind. Werden von vorherigen Programm zusammen mit den Bildern abgespeichert
path_to_image_and_csv_foulder = "C:/Users/printezis_adm/Desktop/animation_2/MOSAIC"

# VIC-3D Projektdatei, in welcher für die Ausgeschnittenen AOI die Startpunktsuche ausgeführt wurde
path_to_input_zip = "C:/Users/printezis_adm/Desktop/animation_2/MOSAIC/animation_2.z3d"

# Ordner, in dem die Bilder sind, die für die finale Auswertung verwendet werden sollen
path_to_real_images = r"C:\Users\printezis_adm\Desktop\animation_2"


# Anzahl an Prozessen, welche die parallel die Berechnung durchfuehren
number_of_processes = 6


def put_nodes_in_Queue(startpoint_node_queue, raw_guess_node_queue):
    while(True):
        startpoint_node = startpoint_node_queue.get()
        if startpoint_node is None:
            return

        for guess_node in startpoint_node:
            raw_guess_node_queue.put(guess_node)


def calculate_start_point_entry(raw_guess_node_queue, edited_guess_node_queue):
    while(True):
        guess_node = raw_guess_node_queue.get()
        if guess_node is None:
            return

        original_p_values = np.array(guess_node.text.split(" ")).astype(np.float64)
        image_name = guess_node.get('img')
        
        df = pd.read_csv(path_to_image_and_csv_foulder + "/" + image_name[0:len(image_name)-4] + ".csv", low_memory=False)

        x = df['x'].to_numpy()
        y = df['y'].to_numpy()

        x_aoi = df['x_aoi'].to_numpy()
        y_aoi = df['y_aoi'].to_numpy() 


        aoi_index = np.argmin(np.sqrt(np.square(x_aoi - original_p_values[0]) + np.square(y_aoi - original_p_values[1])))

        angle = math.radians(df['a'].to_numpy()[aoi_index])

        original_mat = np.array([[original_p_values[5] + 1, original_p_values[4]],
        [original_p_values[3], original_p_values[2] + 1]])

        q1 = np.cos(-angle)
        q2 = -np.sin(-angle)
        q3 = np.sin(-angle) 
        q4 = np.cos(-angle)
        rot_mat = np.array([[q1, q2],
                [q3, q4]])

        sub_mat = np.array([[1, 0],
        [0, 1]])

        final_mat = np.dot(rot_mat, original_mat) - sub_mat

        small_r_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle) , np.cos(angle)]])
        temp_new_xy = np.array([original_p_values[0] - x_aoi[aoi_index], original_p_values[1] - y_aoi[aoi_index]])
        temp_new_xy = np.dot(small_r_mat, temp_new_xy)


        original_p_values[0] = x[aoi_index] + temp_new_xy[0]
        original_p_values[1] = y[aoi_index] + temp_new_xy[1]
        original_p_values[2] = final_mat[1][1]
        original_p_values[3] = final_mat[1][0]
        original_p_values[4] = final_mat[0][1]
        original_p_values[5] = final_mat[0][0]


        guess_node.text = ' '.join(map(str, original_p_values))

        edited_guess_node_queue.put(guess_node)



if __name__ == '__main__':
    start_time = timeit.default_timer()

    startpoint_node_queue = Queue(maxsize=1)
    raw_guess_node_queue = Queue(maxsize=10)
    edited_guess_node_queue = Queue(maxsize=10)   

    put_nodes_in_Queue_process = Process(target=put_nodes_in_Queue, args=(startpoint_node_queue, raw_guess_node_queue))
    list_of_calculate_start_point_node_process = []

    for i in range(number_of_processes):
        list_of_calculate_start_point_node_process.append(Process(target=calculate_start_point_entry, args=(raw_guess_node_queue, edited_guess_node_queue)))


    put_nodes_in_Queue_process.start()

    for i in range(number_of_processes):
        list_of_calculate_start_point_node_process[i].start()
         

    xml_output_path, z3d_name = os.path.split(path_to_input_zip)
    zip_file =  ZipFile(path_to_input_zip, mode="r")
    project_file = zip_file.open('project.xml', mode='r')

    tree = ET.parse(project_file)
    root = tree.getroot()
    root.set("dir", path_to_real_images)


    projectaois_node = root.find('projectaois')


    for startpoint_node in projectaois_node:
        if startpoint_node.tag != "startpoint":
            continue

        print(startpoint_node.tag, startpoint_node.attrib)

        _, ref_image_name = os.path.split(startpoint_node.get('ref_img'))
        startpoint_node.set('ref_img', path_to_real_images + "/" + ref_image_name)
        

        startpoint_node_queue.put(startpoint_node)


        for i in range(len(startpoint_node)):
            startpoint_node[i] = edited_guess_node_queue.get()

    to_remove = []
    for startpoint_node in projectaois_node:
        if startpoint_node.tag == "aoinode":
            to_remove.append(startpoint_node)
            continue

    for aoinode in to_remove:
        projectaois_node.remove(aoinode)


    zip_file.close()



    startpoint_node_queue.put(None)

    for i in range(number_of_processes):
        raw_guess_node_queue.put(None)

    for i in range(number_of_processes):
        list_of_calculate_start_point_node_process[i].join()



    tree.write(path_to_real_images + '/87q4wtho8iusefgoijasdf9oq3erxklsafdbfgoij.xml', encoding='utf-8', xml_declaration=True)
    zip_file = ZipFile(path_to_real_images + "/Result_" + z3d_name, "w")
    zip_file.write(path_to_real_images + '/87q4wtho8iusefgoijasdf9oq3erxklsafdbfgoij.xml', 'project.xml')
    zip_file.close()
    os.remove(path_to_real_images + '/87q4wtho8iusefgoijasdf9oq3erxklsafdbfgoij.xml')

    print("Berechnungszeit: ", str(timeit.default_timer() - start_time))
