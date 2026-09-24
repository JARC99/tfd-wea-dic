import torch
from torch import nn
import torch.nn.functional as F
import glob
import cv2 
import numpy as np
import timeit
import math
import pandas as pd
from multiprocessing import Process, Value, shared_memory, Lock, Condition, Queue
import os
from matplotlib import pyplot as plt


#MODEL_PATH = "D:/Auswertung DIC/Skripte/ROI_NET.pt"
#PATH_TO_IMAGES = "D:/Auswertung DIC/Messung19/*1.tif"
#PATH_TO_IMAGES = "D:/Masterarbeit Jannis/Daenemark 3 YOLO3 - SP1/Messung 3/*0.tif"
#OUTPUT_PATH = "D:/Masterarbeit Jannis/Output/"
#OUTPUT_PATH = "D:/Masterarbeit Jannis/Output"
MODEL_PATH = r"H:\HiWi Cordova\Python\Code_Kiel/ROI_NET_22.04.25MehrParameter.pt"
PATH_TO_IMAGES = r"H:\HiWi Cordova\animation_pitch/*1.tif"
OUTPUT_PATH = r"H:\HiWi Cordova\animation_pitch\mosaic"

# Legt fest, wie gross der ausgeschnittene Bereich sein soll. Pixelgroesse des Bereichs ist: sub_image_size * 2 * (256-2*border)
SUB_IMAGE_SIZE = 1 # 2  1 oder 1.5
NUMBER_OF_READ_PROCESSES = 10
# Ueberlappung der einzelnen Kacheln bei der Segmentierung. Ein zu geringer Wert (ca. < 20) kann dazu fuehren, dass die Segmentierung nicht mehr korrekt funktioniert.
# Wenn der Abstand einer ausseren AOI zum Bildrand jedoch zu gering ist kann es sinnvoll sein, diesen Wert zu verringern.
border = 20

# Legt fest, ob zusaetzlich zu den BTH-Ausschnitten auch noch Ausschnitte ohne Anwendung der BTH gespeichert werden sollen. Diese können verwendet werden, um die Korrelation in VIC-3D durchzufuehren, was 
# bei kleineren Bildmassen schneller geht als bei grossen 
create_additional_images_without_bth = False

# AOI zu klein, um echte AOI zu sein. So werden kleine Strukturen, die faelschlicherweise als AOI erkannt wurden entfernt. Wenn dieser Wert zu klein gewaehlt ist kann das aber auch dazu fueren, 
# dass erkannte AOI verworfen werden.
minimum_AOI_size = 20 

# Tasten zur Aenderung der Helligkeit und des Kontrastes:
#contrast_correction + 0.1  # q
#contrast_correction - 0.1  # w
#brightness_correction + 0.1 # e
#brightness_correction - 0.1  # r



class SharedMemoryImageQueue:
    """
    Klasse, welche aehnliche Schnittstellen wie eine Queue von multiprocessing hat. Wurde allerdings mit SharedMemory implementiert und ist deshalb schneller.
    Wird benutzt, um zwischen den Prozesses Bilder(er) auszutauschen.
    """

    def __init__(self, queue_size, image_shape, dtype=np.uint8):
        """
        Parameter:
            queue_size - gibt an, wie viele Plaetze die Queue hat
            image_shape - Shape der Bild(er). kann also z.B. auch (4,1,256,256) sein
            dtype - Datentype der Daten
        """
        self.queue_size = queue_size
        self.image_shape = image_shape
        self.dtype = dtype
        self.image_size = int(np.prod(image_shape) * np.dtype(dtype).itemsize)
        self.shm = [
            shared_memory.SharedMemory(create=True, size=self.image_size)
            for _ in range(queue_size)
        ]
        self.lock = Lock()
        self.condition = Condition(self.lock)  # Synchronisation
        self.head = Value('i', 0)  # Schreibindex
        self.tail = Value('i', 0)  # Leseindex
        self.count = Value('i', 0)  # Anzahl gespeicherter Elemente
        self.contains_none = [
            Value('i', 0) 
            for _ in range(queue_size)
        ]

    def put(self, image: np.ndarray):
        """
        Packt das Bild in den SharedMemory.
        Parameter:
            image - Daten, die in den SharedMemory gepackt werden. Kann nur ein Bild sein oder auch mehrere auf einmal 
        """
        if image is not None:
            assert image.shape == self.image_shape, "Falsche Bildgröße!"
        with self.condition:
            while self.count.value >= self.queue_size:
                self.condition.wait()  # Warten, falls die Queue voll ist
            shm_buf = np.ndarray(self.image_shape, dtype=self.dtype, buffer=self.shm[self.head.value].buf)
            if image is not None:
                np.copyto(shm_buf, image)  # Bild in Shared Memory kopieren
                self.contains_none[self.head.value].value = 0
            else: # image ist None -> setzte Variable, damit spaeter auch wieder None zurueckgegeben werden kann
                self.contains_none[self.head.value].value = 1
            self.head.value = (self.head.value + 1) % self.queue_size  # Zyklische Bewegung
            self.count.value += 1
            self.condition.notify()  # Weckt wartende Leser auf


    def get(self):
        """
        Gibt das naechste Element in dem SharedMemory zurueck. Es handelt sich dabei NICHT um eine Kopie. D.h. dieser Platz im SharedMemory
        darf weiterhin nicht veraendert werden. So spart man etwas Zeit, da der copy()-Aufruf wegfaellt.
        """
        with self.condition:
            while self.count.value <= 0:
                self.condition.wait()  # Warten, falls die Queue leer ist
            shm_buf = np.ndarray(self.image_shape, dtype=self.dtype, buffer=self.shm[self.tail.value].buf)
            if self.contains_none[self.tail.value].value == 1: # Das Eingabebild fuer die Queue war None. Also gebe None zurueck
                self.tail.value = (self.tail.value + 1) % self.queue_size  # Zyklische Bewegung
                self.count.value -= 1
                self.condition.notify()  # Weckt wartende Schreiber auf       
                return None
            image = shm_buf[:]  # Bild aus Shared Memory lesen
            return image

    def free(self):
        """
        Gibt die Speicherstelle frei in welcher das Bild(er) ist, welches man ueber get erhalten hat. In diese Speicherstelle kann nun ein neues Bild mit put gepackt werden.
        """
        with self.condition:
            self.tail.value = (self.tail.value + 1) % self.queue_size  # Zyklische Bewegung
            self.count.value -= 1
            self.condition.notify()

    def close(self):
        for shm in self.shm:
            shm.close()
            shm.unlink()

    def qsize(self):
        return self.count.value


class AxialDW(nn.Module):
    """
    U-Net
    Paper zur Architektur: https://ieeexplore.ieee.org/document/10317244
    GutHub: https://github.com/duong-db/U-Lite
    """

    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pw = nn.Conv2d(in_c + out_c, out_c,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel = (7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)


    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x
    
class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""
    def __init__(self, dim):
        super().__init__()
        gc = dim//4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw2 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.dw3 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 3)
        self.bn = nn.BatchNorm2d(4*gc)
        self.pw2 = nn.Conv2d(4*gc, dim, kernel_size=1)
        self.act = nn.GELU()


    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x

class ULite(nn.Module):
    def __init__(self):
        super().__init__()
        # Die Anzahl an Parametern in den vorderen Schichten wurde etwas erhöht. Hat fuer die Testdaten etwas besser funktioniert.
        #Encoder
        self.conv_in = nn.Conv2d(1, 64, kernel_size=7, padding='same')
        self.e1 = EncoderBlock(64, 96)
        self.e2 = EncoderBlock(96, 128)
        self.e3 = EncoderBlock(128, 192)
        self.e4 = EncoderBlock(192, 256)
        self.e5 = EncoderBlock(256, 512)

        #Bottle Neck
        self.b5 = BottleNeckBlock(512)

        #Decoder
        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 192)
        self.d3 = DecoderBlock(192, 128)
        self.d2 = DecoderBlock(128, 96)
        self.d1 = DecoderBlock(96, 64)

        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)

        # Original Netzaritektur (Andere Parameteranzahl)
        """
        #Encoder
        self.conv_in = nn.Conv2d(1, 16, kernel_size=7, padding='same')
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        #Bottle Neck
        self.b5 = BottleNeckBlock(512)

        #Decoder
        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)

        self.conv_out = nn.Conv2d(16, 1, kernel_size=1)
        """

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)        # (512, 8, 8)

        """Decoder"""
        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = F.sigmoid(self.conv_out(x))
        #x = self.conv_out(x)
        return x

def rotateImage(image, angle):
    """
    Funktion zum Drehen der AOI (gegen den Uhrzeigersinn)
    Parameter:
      image - Bilder der AOI 
      angle - Rotationswinkel
    """
    height, width = image.shape
    center = tuple(np.array([width, height]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (width, height))
    return new_image

def load_images(image_path_queue, contrast_image_queue, image_queue, current_image_number, brightness_correction, contrast_correction, condition):
    """
    Funktion welche von mehreren Prozessen verwendet wird, um die Bilder zu laden
    Parameter:
      image_path_queue - Queue, welche die einzelnen Pfade fuer die Bilder enthaelt
      contrast_image_queue - SharedMemory, in welche die Bilder gepackt werden, nachdem Helligkeit und Kontrast passend fuer das Netz angepasst wurden 
      image_queue - SharedMemory, in welche die Originalbilder gepackt werden
      current_image_number - geteilte Variable ueber welche kommuniziert wird, welches Bild als naechstes in die Queue gepack werden muss
      brightness_correction - Anpassungswert fuer "Helligkeit" des Bildes
      contrast_correction - Anpassungswert fuer Kontrast des Bildes
      condition - Condition zum pausieren der Leseprozesse, wenn diese ihr Bild schon geladen haben aber es wegen der Reihenfolge noch nicht in die Queue legen koennen
    """
    while(True):
        paths_to_image, image_number = image_path_queue.get()
        if paths_to_image is None: # Der Prozess kann beendet werden. 
            while (current_image_number.value != image_number):
                pass
            contrast_image_queue.put(None) # zum Signalisieren der weiteren Prozesse, dass keine neuen Bilder mehr folgen
            image_queue.put(None)
            with current_image_number.get_lock():
                current_image_number.value += 1
            break

        image = cv2.imread(paths_to_image, -1) 
        contrast_image = cv2.addWeighted( image, contrast_correction, image, 0, brightness_correction)
        contrast_image = contrast_image.astype(np.uint64)
        
        with condition:
            while (current_image_number.value != image_number): # noch nicht dran mit Bild in Queue packen, da Reihenfolge sonst falsch
                condition.wait()
            
            contrast_image_queue.put(contrast_image)
            image_queue.put(image)
            with current_image_number.get_lock():
                current_image_number.value += 1
            condition.notify_all()



def get_AOI_coordinate(event,x,y,flags,param):
    """
    Gibt die Koordinaten zurueck, auf welche der Benutzer geklickt hat. Wird verwendet wenn der Benutzer dem Programm helfen muss eine AOI zu finden, wenn diese nicht automatisch gefunden wurde
    """
    if event == cv2.EVENT_LBUTTONDBLCLK:
        get_AOI_coordinate.coordinate = (x,y)
        cv2.destroyAllWindows() 
    else:
        get_AOI_coordinate.coordinate = None
    
        
def calculate_net(shared_memory, lock_start_net_calculation, lock_take_net_results):
    """
    Berechnet die Masken mit dem Netz auf der GPU. 
    Parameter:
      shared_memory - SharedMemory, in welchen sowohl die Kacheln der AOI gepackt werden, sowie nach der Segmentierung die Masken. Die Originalkacheln werden dabei ueberschrieben
      lock_start_net_calculation - Lock, wenn dieser freigegeben wird beginnt die Berechnung des Netzes auf der GPU
      lock_take_net_results - Lock, durch die Freigabe wird signalisiert, dass die Masken fuer die Kacheln fertig berechnet worden sind. 
    """
    model = ULite()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Device: ", device)


    while(True):
        lock_start_net_calculation.acquire() # warte auf Freigabe zur Berechnung des Netzes.Andernfalls steht im SharedMemory noch kein valider Wert
        aoi_segment_np_array = shared_memory.get()
        if aoi_segment_np_array is None:
            break
        tensor_on_gpu = torch.tensor(aoi_segment_np_array, device=device)
        shared_memory.free()
        result_on_gpu = model(tensor_on_gpu)
        result = result_on_gpu.detach().cpu().numpy()
        shared_memory.put(result)
        torch.cuda.synchronize()
        lock_take_net_results.release() # Signalisiere, dass im SharedMemory nun die fertigen Masken sind
            


def calculate_segmentation_mask(net_shared_memory, contrast_image_queue, mask_queue, original_aoi_pos_list, start_counter, lock_start_net_calculation, lock_take_net_results):
    """
    Fuehrt die Vor- und Nachbereitung fuer die Maskenberechnung auf der GPU durch. Berechnet die Kacheln der AOI und die neue Position der AOI. 
    Parameter:
      net_shared_memory - SharedMemory, in welchen sowohl die Kacheln der AOI gepackt werden, sowie nach der Segmentierung die Masken. Die Originalkacheln werden dabei ueberschrieben.
      contrast_image_queue - Die angepassten Messbilder. 
      mask_queue - Enthalt eine Liste der Masken fuer die AOI fuer ein Messbild sowie die Positionen der AOI. Aussehen: ([Masken], [Positionen])
      original_aoi_pos_list - Die Positionen der AOI, welche der Benutzer zu Beginn markiert hat
      start_counter - Gibt an wie viele Bilder benutzt werden, um zu Beginn den Kreis zur Bestimmung des Rotationswinkels zu bestimmen. Nachdem der Counter abgelaufen ist beginnt die 
                    Berechnung von Vorne und die Minimierung der Rotordrehnung wird ausgefuehrt
      lock_start_net_calculation - Lock, wenn dieser freigegeben wird beginnt die Berechnung des Netzes auf der GPU
      lock_take_net_results - Lock, durch die Freigabe wird signalisiert, dass die Masken fuer die Kacheln fertig berechnet worden sind. 
    """


    # Enthalt die Bildnummer und die Position fuer jede AOI. Nach der Bearbeitung einer AOI wird die neue Position abgespeichert
    # und die Bildnummer wird um 1 erhoeht
    bearbeitungsliste = []
    for i, p in enumerate(original_aoi_pos_list):
        bearbeitungsliste.append((0, p, p))


    # Enthaelt Informationen zu der AOI, welche gerade auf der PGU ist bzw. als letztes dort berechnet wurde.
    # Enthaelt:
    # - Die Bildnummer, von welchem die AIO stammte
    # - Die vermutete Position der AOI, welche im vorherigen Durchlauf berechnet wurde
    # - Die Position oberste linke des AOI-Ausschnittes im Messbild
    aoi_on_gpu = None
    aoi_was_on_gpu = None

    # Enthaelt die einzelnen Masken der AIO als Liste pro Bild
    mask_list = []

    # Enthaelt die einzelnen Positionen der AOI im Bild
    coordinates_sorted_by_aoi = []


    # Signalisiert, dass noch weitere AOI-Ausschnitte folgen werden. Wenn False folgen keine weiteren AOI-Ausschnitte mehr. Dann muessen z.B.
    # keine neuen Bilder mehr auf die GPU geladen werden. Nur das aktuelle Bild auf der GPU muss noch gelesen werden
    is_not_last_AOI = True

    image_shape = None
    image = None
    window_name = "Mittelpunkt AOI markieren"


    # Maskenbild fuer die AOI, die gerade berechnet wird
    mask = np.full((SUB_IMAGE_SIZE * (256-2*border) * 2, SUB_IMAGE_SIZE * (256-2*border) * 2), 0, dtype=np.uint16)
    aoi_segment_np_array = np.empty(shape=((SUB_IMAGE_SIZE * 2)**2, 1,256,256), dtype = np.float32)
    kernel_for_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
    image_number = -1
    number_of_not_found_aoi = 0


    while(True):
        work_image_number, position, old_pos = bearbeitungsliste.pop(0)

        if image_number < work_image_number:
            if image is not None:
                contrast_image_queue.free()
            image = contrast_image_queue.get()
            image_number += 1

            if image_shape is None:
                image_shape = image.shape
            print("###################################################")
            print("Eingangswarteschlange: ", contrast_image_queue.qsize())
            print("Ausgangswarteschlange: ", mask_queue.qsize())
            print("Bildnummer: ", str(image_number))
            print("Anzahl nicht gefundene AOI: ", str(number_of_not_found_aoi))


            # Es gibt keine weiteren Messbilder mehr. Es gibt also nur noch auf der GPU eine AOI, die noch verarbeitet fertig verarbeitet werden muss
            if image is None:
                is_not_last_AOI = False

        if is_not_last_AOI:
            x_min = max(position[1] - int((256-2*border) * SUB_IMAGE_SIZE), border)
            x_max = min(position[1] + int((256-2*border) * SUB_IMAGE_SIZE), image_shape[0] - border)
            y_min = max(position[0] - int((256-2*border) * SUB_IMAGE_SIZE), border)
            y_max = min(position[0] + int((256-2*border) * SUB_IMAGE_SIZE), image_shape[1] - border)
            x_min_diff = (x_min - (position[1] - int((256-2*border) * SUB_IMAGE_SIZE)))
            y_min_diff = (y_min - (position[0] - int((256-2*border) * SUB_IMAGE_SIZE)))
            x_max_diff = - (position[1] + int((256-2*border) * SUB_IMAGE_SIZE) - x_max)
            y_max_diff = - (position[0] + int((256-2*border) * SUB_IMAGE_SIZE) - y_max)


            # schneide aus dem Messbild die Kacheln aus, in denen die AOI vermutet wird. Diese Kacheln werden fuer die Segmentierung benutzt
            counter = 0
            for x in range(x_min + x_max_diff, x_max + x_min_diff, (256-2*border)):
                for y in range(y_min + y_max_diff, y_max + y_min_diff, (256-2*border)):
                    aoi_segment_np_array[counter] = [image[x-border:x+(256-border), y-border:y+(256-border)] / (255 * 255)]
                    counter += 1


            # Wenn noch keine Daten auf der GPU sind, dann stelle die ersten Daten zur Verfuegung, die auf die GPU gepackt werden sollen
            # Ergebnisse muessen noch nicht abgefolt werden, deshab danach continue
            if aoi_on_gpu is None:
                aoi_on_gpu = (work_image_number, position, (y_min, x_min), old_pos, image)
                net_shared_memory.put(aoi_segment_np_array)
                lock_start_net_calculation.release()
                continue
            

        # Ergebnis vom Netz holen, wenn das Netz mit der Berechnung fertig ist
        lock_take_net_results.acquire()
        result = net_shared_memory.get().copy()
        net_shared_memory.free()


        aoi_was_on_gpu = aoi_on_gpu
        if is_not_last_AOI: # Wenn es nicht die letzte AOI in der gesamten Messung ist, dann stelle die naechste AOI zur Verfuegung, um diese auf der GPU berechnen zu lassen
            aoi_on_gpu = (work_image_number , position, (y_min, x_min), old_pos, image)
            net_shared_memory.put(aoi_segment_np_array)
            lock_start_net_calculation.release()
            
        # setze die einzelnen Kacheln zu einem Maskenbild fuer die AOI zusammen. Die aeusseren Raender der Kacheln werden verworfen.
        result = result[:, 0, border : (256-border), border : (256-border)]
        result = (result * 255 * 255).astype(np.uint16)
        counter = 0
        for x in range(0, SUB_IMAGE_SIZE * (256-2*border) * 2, (256-2*border)):
            for y in range(0, SUB_IMAGE_SIZE * (256-2*border) * 2, (256-2*border)):
                mask[x : x + (256-2*border), y : y + (256-2*border)] = result[counter, :, :]
                counter += 1


        # Das Maskenbild der AOI wird herunterskaliert, um moeglichst schnell alle zusammenhaengenden Komponenten zu finden.
        # Mehr als eine zusammenhaegende Komponente bedeutet, dass zwei AOI im Bildausschnitt sind. Es wird 
        # spaeter die Maske ausgewaehlt, die am naechsten zur vorhergesagten Position ist
        small_mask = cv2.resize(mask, (0, 0), fx=0.2, fy=0.2)
        threshold = cv2.threshold(small_mask, 128 * 255, 255, cv2.THRESH_BINARY)[1]
        number_of_labels, labels = cv2.connectedComponents(threshold.astype(np.uint8))


        mask_of_component_list = []
        coordinates = []
        coordinates_in_AOI_image = []
        for label in range(1,number_of_labels):
            points = np.argwhere(labels == label)
            if (len(points) < minimum_AOI_size):
                continue            
            temp_mask = np.zeros(shape=small_mask.shape, dtype=np.uint16)
            temp_mask[labels == label] = 255 * 255
            mask_of_component_list.append(temp_mask)
            coordinate = np.mean(points, axis=0, dtype=int)
            global_coordinate = (aoi_was_on_gpu[2][0] + coordinate[1] * 5, aoi_was_on_gpu[2][1] + coordinate[0] * 5)
            x_min = max(global_coordinate[1] - int((256-2*border) * SUB_IMAGE_SIZE),0)
            x_max = min(global_coordinate[1] + int((256-2*border) * SUB_IMAGE_SIZE), image_shape[0])
            y_min = max(global_coordinate[0] - int((256-2*border) * SUB_IMAGE_SIZE), 0)
            y_max = min(global_coordinate[0] + int((256-2*border) * SUB_IMAGE_SIZE), image_shape[1])
            x_min_diff = (x_min - (global_coordinate[1] - int((256-2*border) * SUB_IMAGE_SIZE)))
            y_min_diff = (y_min - (global_coordinate[0] - int((256-2*border) * SUB_IMAGE_SIZE)))
            x_max_diff = - (global_coordinate[1] + int((256-2*border) * SUB_IMAGE_SIZE) - x_max)
            y_max_diff = - (global_coordinate[0] + int((256-2*border) * SUB_IMAGE_SIZE) - y_max)
   
            coordinates.append((global_coordinate[0] + y_min_diff + y_max_diff, global_coordinate[1] + x_min_diff + x_max_diff))
            coordinates_in_AOI_image.append((coordinate[1] * 5 + y_min_diff + y_max_diff, coordinate[0] * 5 + x_min_diff + x_max_diff))    


        # Von allen gefundenen Komponenten (Masken) waehle die aus, die am nachsten an der vorher berechneten neuen Position ist
        min_index = 0 
        min_distance = np.inf
        for ii, c in enumerate(coordinates):
            if ((aoi_was_on_gpu[1][0] - c[0])**2 + (aoi_was_on_gpu[1][1] - c[1])**2)**(0.5) < min_distance:
                min_distance = ((aoi_was_on_gpu[1][0] - c[0])**2 + (aoi_was_on_gpu[1][1] - c[1])**2)**(0.5)
                min_index = ii

        if len(mask_of_component_list) > 0: # Mindestens eine AOI gefunden
            # Entferne andere Masken von AOIs, die nicht im Mittelpunkt stehen. Es bleibt nur die Maske der AOI, die am naechsten an der neuen berechneten Position der AOI ist
            temp_mask = cv2.morphologyEx(mask_of_component_list[min_index], cv2.MORPH_DILATE, kernel_for_dilate)
            temp_mask = cv2.resize(temp_mask, mask.shape)
            temp_mask = np.minimum(temp_mask, mask)  


            # Verschiebe das Maskenbild so, dass die AIO genau in der Mitte ist      
            translation_matrix = np.float32([ [1,0,SUB_IMAGE_SIZE * (256-2*border) - coordinates_in_AOI_image[min_index][0]]  , [0,1, SUB_IMAGE_SIZE * (256-2*border) - coordinates_in_AOI_image[min_index][1]]])
            temp_mask = cv2.warpAffine(temp_mask, translation_matrix, temp_mask.shape)
            # Die neue Position der AIO im naechsten Bild wird grob berechnet
            p0 = coordinates[min_index][0] + (coordinates[min_index][0] - aoi_was_on_gpu[3][0])
            p1 = coordinates[min_index][1] + (coordinates[min_index][1] - aoi_was_on_gpu[3][1])
            last_position = coordinates[min_index]
        else: # AOI nicht gefunden
            number_of_not_found_aoi += 1
            temp_mask = np.full(fill_value=255 * 255, shape=mask.shape, dtype=np.uint16)
            ret = None
            image_with_dot = aoi_was_on_gpu[4].copy()
            image_with_dot = cv2.cvtColor(image_with_dot,cv2.COLOR_GRAY2RGB)
            cv2.circle(image_with_dot,(aoi_was_on_gpu[1][0],aoi_was_on_gpu[1][1]),75,(0, 0, 255 * 255),-1)
            while ret is None:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, image_with_dot) 
                cv2.setMouseCallback(window_name, get_AOI_coordinate) 
                cv2.waitKey(0) & 0xFF
                ret = get_AOI_coordinate.coordinate
            p0 = ret[0] + (ret[0] - aoi_was_on_gpu[3][0])
            p1 = ret[1] + (ret[1] - aoi_was_on_gpu[3][1])
            coordinates.append([ret[0], ret[1]])
            last_position = (ret[0], ret[1])


        coordinates_sorted_by_aoi.append(coordinates[min_index])
        mask_list.append(temp_mask)
        bearbeitungsliste.append((aoi_was_on_gpu[0] + 1, (p0, p1), last_position))


        # Fuer alle AOI in einem Bild wurden alle Masken berechnet. Damit kann die Liste mit berechneten Masken und berechneten Positionen an den naechsten Prozess uebergeben werden
        if len(mask_list) == len(original_aoi_pos_list):
            mask_queue.put((mask_list, coordinates_sorted_by_aoi))
            mask_list = []
            coordinates_sorted_by_aoi = []
            if start_counter > 0:
                start_counter -= 1
                if start_counter == 0:
                    bearbeitungsliste = []
                    for p in original_aoi_pos_list:
                        bearbeitungsliste.append((aoi_was_on_gpu[0] + 1, p, p))
                    aoi_on_gpu = None
                    aoi_was_on_gpu = None
                    # Verwerfen des Bildes, welche gerade auf der GPU berechnet wird, da der Rotationswinkel der AOI nun berechnet werden kann und die Messung von Vorne beginnt
                    lock_take_net_results.acquire()
                    net_shared_memory.get()
                    net_shared_memory.free()


        # Die letzte AOI wurde berechnet. Den folgenden Prozessen wird ueber "None" in den Queues signalisiert, dass 
        # keine weiteren Daten mehr folgen
        if not is_not_last_AOI:
            mask_queue.put((None, None))
            net_shared_memory.put(None)
            lock_start_net_calculation.release()
            break



def calculate_tiles(image_queue, mask_queue, final_images_queue, raster_size, AOI_size_in_raster, index_of_outer_aoi, start_counter):
    """
    Segmentiert mithilfe der Masken (aus mask_queue) die AOI im Originalbild (image_queue), fuehrte die BTH-Transformation durch und minimiert die Rotation
    Parameter:
      image_queue - SharedMemory mit dem Originalmessbildern. Aus diesen wird mithilfe der Masken (mask_queue) die AOI segmentiert
      mask_queue - Queue, in welcher sich eine Liste mit Maskenbildern und eine Liste mit den Positionen der AOI im Originalbild befindet
      final_images_queue - Queue, besteht aus: AIO-Zwischenbild und Pandas-Datarame
      raster_size - Zwischenbild Hoehe x Breite (beides hat die Einheit AOI)
      AOI_size_in_raster - Die Groesse einer AOI im Zwischenbild 
      index_of_outer_aoi - Indexe der AOI, die fuer die Kreisberechnung des Rotors benutzt werden
      start_counter - Gibt an wie viele Bilder benutzt werden, um zu Beginn den Kreis zur Bestimmung des Rotationswinkels zu bestimmen
    """
    angle = 0
    aoi_raster = np.full((int(AOI_size_in_raster * raster_size), int(AOI_size_in_raster * raster_size)), 0, dtype=np.uint16)
    aoi_raster_BTH = np.full((int(AOI_size_in_raster * raster_size), int(AOI_size_in_raster * raster_size)), 0, dtype=np.uint16)
    point_for_ellipse = np.empty((0, 2), int)
    image_number = 0
    image_number_for_naming = 0


    original_image = None
    Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8)) 
    KernelG = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)) 
    eccentricity1 = []
    eccentricity2 = []
    eccentricity3 = []


    while(True):
        if original_image is not None:
            image_queue.free()
        original_image = image_queue.get()
        mask_list, aoi_pos_list = mask_queue.get()

        if original_image is None:
            final_images_queue.put((None, None, None))
            df2 = pd.DataFrame({ 'a': eccentricity1,
            'b': eccentricity2,
            'eccentricity': eccentricity3})
            df2.to_csv(OUTPUT_PATH + "/Exzentrizitat.csv", index=False, decimal=',', sep=';')  
            break

        # Variablen fuer die CSV-Datei
        old_x_position_list = []
        old_y_position_list = [] 
        rotation_angle_list = []
        x_position_in_AIO_list = []
        y_position_in_AIO_list = []   


        for index, p in enumerate(aoi_pos_list):

            # Berechne das finale Fenster fuer die AIO:
            x_min = p[1] - int((256-2*border) * SUB_IMAGE_SIZE)
            x_max = p[1] + int((256-2*border) * SUB_IMAGE_SIZE)
            y_min = p[0] - int((256-2*border) * SUB_IMAGE_SIZE)
            y_max = p[0] + int((256-2*border) * SUB_IMAGE_SIZE)
    
            secmentated_aoi = np.minimum(original_image[x_min : x_max , y_min : y_max ], mask_list[index]) 
            secmentated_aoi_strong_blured = cv2.blur(secmentated_aoi, (5,5), cv2.BORDER_DEFAULT)
            secmentated_aoi_strong_blured = cv2.blur(secmentated_aoi_strong_blured, (5,5), cv2.BORDER_DEFAULT)
            secmentated_aoi_dilate = cv2.morphologyEx(secmentated_aoi_strong_blured, cv2.MORPH_DILATE, KernelG)
            
            invertierte_maske = np.bitwise_not(mask_list[index])
            rand_dilated = np.minimum(invertierte_maske, secmentated_aoi_dilate)
            both = np.maximum(secmentated_aoi, rand_dilated)
            both = cv2.blur(both, (3,3), cv2.BORDER_DEFAULT)
            tophat_image = cv2.morphologyEx(both, cv2.MORPH_BLACKHAT, Kernel)
            aoi_image_BTH = np.minimum(tophat_image, mask_list[index])
           

            aoi_image_BTH = rotateImage(aoi_image_BTH, angle)

            if create_additional_images_without_bth:
                aoi_image = original_image[x_min : x_max , y_min : y_max ]
                aoi_image = rotateImage(aoi_image, angle)
                
            # Berechne Rasterkoordinate:
            raster_start_y = int(index / raster_size) * AOI_size_in_raster
            raster_start_x = (index % raster_size) * AOI_size_in_raster

            # Fuege AIO im Zwischenbild ein:
            aoi_raster_BTH[raster_start_y:raster_start_y+AOI_size_in_raster, raster_start_x:raster_start_x+AOI_size_in_raster] = aoi_image_BTH

            if create_additional_images_without_bth:
                aoi_raster[raster_start_y:raster_start_y+AOI_size_in_raster, raster_start_x:raster_start_x+AOI_size_in_raster] = aoi_image


            # Informationen in die Liste fuer die CSV-Datei packen
            old_x_position_list.append(p[0])
            old_y_position_list.append(p[1])
            x_position_in_AIO_list.append(raster_start_x + int(AOI_size_in_raster / 2))
            y_position_in_AIO_list.append(raster_start_y + int(AOI_size_in_raster / 2))
            rotation_angle_list.append(angle)


            # Wenn AIO aeussere AIO ist, dass packe Koordinaten in die Liste zur Winkelbestimmung. Nur jeder 3te Punkt wird im normalen Ablauf (also nicht am Anfang),
            # damit Liste nicht zu lang wird. Wenn Liste zu voll ist werden die ersten Werte wieder entfernt
            if index in index_of_outer_aoi and (start_counter >  0 or image_number % 3 == 0):
                point_for_ellipse = np.append(point_for_ellipse, np.array([p[0:2]]), axis=0)
                if len(point_for_ellipse) > 60:
                    point_for_ellipse = point_for_ellipse[len(point_for_ellipse) - 60:len(point_for_ellipse)]


        if start_counter > 0:    # Anzahl an Bildern zur Ermittlung des Kreises wurde noch nicht erreicht
            start_counter -= 1


        if start_counter <=  0: # Anzahl an Bildern zur Ermittlung des Kreises wurde erreicht und Rotationswinkel kann nun bestimmt werden
            ellipse = cv2.fitEllipse(point_for_ellipse)
            point_on_ellipse = np.array(aoi_pos_list[index_of_outer_aoi[1]][0:2]) - np.array([ellipse[0][0], ellipse[0][1]])
            local_eccentricity = math.sqrt(max(ellipse[1][0] / 2, ellipse[1][1] / 2)**2 - min(ellipse[1][0] / 2, ellipse[1][1] / 2)**2) / max(ellipse[1][0] / 2, ellipse[1][1] / 2)
            eccentricity1.append(max(ellipse[1][0], ellipse[1][1]) / 2)
            eccentricity2.append(min(ellipse[1][0], ellipse[1][1]) / 2)
            eccentricity3.append(local_eccentricity)
            
            # Rotation und Verzerrung der Ellipse entfernen, damit anhand des entstandenen Kreises der WInkel bestimmt werden kann
            ellipse_rotation = math.radians(-ellipse[2])
            rot_mat = np.array([[np.cos(ellipse_rotation), - np.sin(ellipse_rotation)], 
                                [np.sin(ellipse_rotation), np.cos(ellipse_rotation)]])
            
            # Punkt auf Kreis bestimmen
            new_point_on_circle = np.dot(rot_mat, point_on_ellipse.T).T
            new_point_on_circle = new_point_on_circle / ellipse[1]


            # Winkel berechnen
            new_point_on_circle = new_point_on_circle / np.linalg.norm(new_point_on_circle)
            x0 = new_point_on_circle[0]
            x1 = new_point_on_circle[1]
            if x0 > 0:
                angle = math.degrees(math.acos(x1))
            else:
                angle = 360 - math.degrees(math.acos(x1))
            angle = 360 - angle
            if start_counter > -2: # Winkel Offset des ersten Bildes speichern Nummerierung der abgespeicherten Bilder von vorne beginnen
                angle_offset = angle
                image_number_for_naming = 0
                start_counter-= 1
            angle = angle - angle_offset
                
        df = pd.DataFrame({ 'x': old_x_position_list,
                            'y': old_y_position_list,
                            'a': rotation_angle_list, 
                            'x_aoi': x_position_in_AIO_list, 
                            'y_aoi': y_position_in_AIO_list})


        if create_additional_images_without_bth:
            final_images_queue.put((aoi_raster_BTH , aoi_raster, df))
        else:
            final_images_queue.put((aoi_raster_BTH , None, df))

        image_number += 1
        image_number_for_naming += 1


def save_files(final_images_queue, path, file_name_queue):
    """
    Speichert das Bild mit den segmentierten AOI und die zugehoerige CSV-Datei ab
    Parameter:
      final_images_queue - Queue, besteht aus: AIO-Zwischenbild, Pandas-Datarame und der Bildnummer
      path - Pfad zum Ordner, in welchem die Bilder und CSV-Datein abgespeichert werden sollen
      file_name_queue - Queue, welche die original Bildnamen enthaelt
    """
    if not os.path.isdir(path + "/without_BTH/") and create_additional_images_without_bth:
        os.mkdir(path + "/without_BTH/") 
    while(True):
        image_BTH, image, data_frame  = final_images_queue.get()
        if image_BTH is None: # Berechnung abgeschlossen, verlasse schleife und beende so den Prozess
            break

        image_name = file_name_queue.get()
        cv2.imwrite(path + "/" + image_name, image_BTH) 
        if create_additional_images_without_bth:
            image = np.floor_divide(image,265)
            image = image.astype(np.uint8)
            cv2.imwrite(path + "/without_BTH/" + image_name, image) 
        data_frame.to_csv(path + "/" + image_name[0:-4] + ".csv", index=False)  





if __name__ == '__main__':
    aoi_pos_list = []
    index_of_outer_aoi = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    brightness_correction = 1
    contrast_correction = 1
    print("Image path: ", glob.glob(PATH_TO_IMAGES)[0])
    original_image = cv2.imread(glob.glob(PATH_TO_IMAGES)[0], cv2.IMREAD_GRAYSCALE) 

    window_name = "brightness and contrast correction"

    # Loop zum anpassen des Kontrastes und der Helligkeit. Dies kannn benutzt werden, um die Messbilder mehr nach den Trainingsbildern aussehen zu lassen. Dies kann z.B. der Fall sein,
    # wenn es Messbilder gibt, deren Helligkeit oder Kontrast so nicht in den Trainingsdaten vorhanden ist
    ret = 255
    while( ret in [255, 113, 119, 101, 114]):
        print("---> ", original_image.dtype)
        image = cv2.addWeighted( original_image, contrast_correction, original_image, 0, brightness_correction)
        image = image.astype(np.uint8)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image) 

        ret = cv2.waitKey(0) & 0xFF
        if ret == 113: 
            contrast_correction += 0.1  # q
        if ret == 119:
            contrast_correction -= 0.1  # w
        if ret == 101:
            brightness_correction += 0.1 # e
        if ret == 114:
            brightness_correction -= 0.1  # r
        if ret not in [255, 113, 119, 101, 114]:
            break
    cv2.destroyAllWindows() 

    # Funktion welche bei einem Doppel-Mausklick aufgerufen wird. Malt einen Punkt an die Stelle, an welche der Benutzer geklickt hat
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(image,(x,y),50,(255,0,0),-1)
            aoi_pos_list.append((x,y))
            if window_name == "Mark outer AIO":
                index_of_outer_aoi.append(len(aoi_pos_list)-1)
            cv2.destroyWindow(window_name)


    # Loop zum Markieren der aeusseren AIO. Werden fuer die Ermittlung der Rotation gesondert gebraucht
    window_name = "Mark outer AIO"
    ret = 255
    while( ret == 255):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image) 
        cv2.setMouseCallback(window_name, draw_circle) 
        ret = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows() 


    # Loop zum Markieren der restlichen AIO
    window_name = "Mark inner AIO"
    ret = 255
    while( ret == 255):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image) 
        cv2.setMouseCallback(window_name, draw_circle) 
        ret = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows() 

    image_paths = glob.glob(PATH_TO_IMAGES)

    # Berechne AIO Zwischenbild (Hoehe x Breite)
    raster_size = math.ceil(math.sqrt(len(aoi_pos_list)))
    AOI_size_in_raster = int((SUB_IMAGE_SIZE * 2) * (256-2*border))


    # Legt fest wie viele Bilder verarbeitet werden muessen, bevor der Winkel bestimmt werden kann. 
    # Wenn zu wenig Punkte vorhanden sind funktioniert das nicht. Wenn genug Punkte vorhanden sind beginnt 
    # die Auswertung neu, nun aber mit Rotationsanpassung
    start_counter = 10
    for i in range(min(start_counter, len(image_paths))):
        image_paths.insert(i, image_paths[i + i])

    # Aktuelle Bildnummer. Wird von mehreren Prozesses verwendet, die die Bilder laden.
    # Durch die geteilte Variable wird sichergestellt, dass die Bilder in der richtigen Reihenfolge bearbeitet werden
    current_image_number = Value('d', 0.0)
    

    # Die beiden Locks werden verwendet, um zu Signalisieren, wann die Daten auf der GPU berechnet werden sollen und wann diese 
    # fertig berechnet sind
    lock_start_net_calculation = Lock()
    lock_start_net_calculation.acquire()
    lock_take_net_results = Lock()
    lock_take_net_results.acquire()

    # Einhaelt die einzelnen Pfade zu den Bildern. None signalisiert, dass alle Bilder gepaden worden sind
    image_path_queue = Queue(maxsize=15)

    example_image = cv2.imread(glob.glob(PATH_TO_IMAGES)[0], -1) 
 
    # Shared Memory zum schnellen AUstauschen der Bilder zwischen den Prozessen
    contrast_image_queue = SharedMemoryImageQueue(10, example_image.shape, example_image.dtype)
    image_queue = SharedMemoryImageQueue(9, example_image.shape, example_image.dtype)


    # Shared Memory um Kacheln der AOI zur Segmentierung dem Prozess zu uebergeben, welcher diese auf der GPU berechnen laesst. 
    # Berechnete Masken werden danach in diesen SharedMemory gelasen. 
    net_queue = SharedMemoryImageQueue(1, ((SUB_IMAGE_SIZE*2)**2, 1, 256, 256), np.float32)
    final_images_queue = Queue(maxsize=5)
    mask_queue = Queue(maxsize=5)
    file_name_queue = Queue(maxsize=10)
    
    # Dient zur Synchronisation zwischen den Leseprozessen. 
    image_load_condition = Condition(Lock())

    # Erstelle Leseprozesse
    load_images_process_list = []
    for _ in range(NUMBER_OF_READ_PROCESSES):
        load_images_process_list.append(Process(target=load_images, args=(image_path_queue, contrast_image_queue, image_queue, current_image_number, brightness_correction, contrast_correction, image_load_condition)))
                                                      
    # Erstelle weitere verwendete Prozesse                                                        
    net_process = Process(target=calculate_net, args=(net_queue, lock_start_net_calculation, lock_take_net_results))                                                                                                      
    segmentation_process = Process(target=calculate_segmentation_mask, args=(net_queue, contrast_image_queue, mask_queue, aoi_pos_list, start_counter, lock_start_net_calculation, lock_take_net_results))
    tiles_process = Process(target=calculate_tiles, args=(image_queue, mask_queue, final_images_queue, raster_size, AOI_size_in_raster, index_of_outer_aoi, start_counter))
    save_files_process = Process(target=save_files, args=(final_images_queue, OUTPUT_PATH, file_name_queue))

    # Prozesse starten...
    for p in load_images_process_list:
        p.start()


    net_process.start()
    segmentation_process.start()
    tiles_process.start()
    save_files_process.start()

    start_time = timeit.default_timer()

    # Dateinamen(Pfad) in die Queue packen. Wenn alle drin sind noch None hinzufuegen, um alle Leseprozesse zu beenden
    for image_number, paths_to_image in enumerate(image_paths):
        image_path_queue.put((paths_to_image, image_number))
        head, tail = os.path.split(paths_to_image)
        file_name_queue.put(tail)
    for i in range(NUMBER_OF_READ_PROCESSES):
        image_path_queue.put((None, image_number + 1 + i))


    print("Warte auf Beenden")
    for p in load_images_process_list:
        p.join()


    print("load_images_process1", " beendet")

    net_process.join()
    print("net_process", " beendet")
    segmentation_process.join()
    print("segmentation_process", " beendet")
    tiles_process.join()
    print("tiles_process", " beendet")
    save_files_process.join()
    print("save_files_process", " beendet")
    print("Berechnungszeit: ", str(timeit.default_timer() - start_time))