import cv2
import pickle
import cvzone
import numpy as np
import time
import os
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8m.pt') # Puedes cambiar a yolov8m.pt o yolov8l.pt para modelos más grandes


# Obtener el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas para los archivos de posiciones y estado de espacios
car_park_pos_filename = os.path.join(script_dir, 'CarParkPos.pkl')  # Archivo con coordenadas de los espacios
spaces_status_filename = os.path.join(script_dir, 'spaces_status.pkl')  # Archivo con el estado de los espacios

# Cargar la lista de posiciones de los espacios de estacionamiento desde el archivo 'CarParkPos'
try:
    with open(car_park_pos_filename, 'rb') as f:
        posList = pickle.load(f)  # Cargar las posiciones seleccionadas en SpacePicker
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{car_park_pos_filename}'. Asegúrate de que el archivo exista en el directorio actual.")
    exit()
print(posList)


# Crear un diccionario para almacenar el tiempo de ocupación de cada espacio
occupied_times = {tuple(map(tuple, pos)): 0 for pos in posList}

start_times = {tuple(map(tuple, pos)): None for pos in posList}

# Función para verificar los espacios y detectar autos usando YOLO
def checkSpaces(img):
    spaces = 0
    results = model(img, conf=0.25)  # Ajusta el umbral de confianza
         # Usar YOLOv8 para la detección de autos en la imagen

    for pos in posList:
        # Convertir la posición actual en numpy array para formar un polígono
        polygon = np.array(pos, np.int32)
        polygon = polygon.reshape((-1, 1, 2))

        # Crear una máscara negra del tamaño de la imagen
        mask = np.zeros(img.shape[:2], np.uint8)

        # Dibujar el polígono en la máscara
        cv2.fillPoly(mask, [polygon], 255)

        # Detectar si algún auto está dentro del espacio usando las detecciones de YOLO
        detected_car = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])  # Clase detectada
                if cls == 2:  # Clase 'cars' en COCO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas del auto
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Verificar si el auto está dentro del espacio de estacionamiento
                    if cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0:
                        detected_car = True

        # Convertir la posición actual a tupla para usarla como clave
        pos_tuple = tuple(map(tuple, pos))

        if detected_car:  # Si se detecta un auto
            color = (0, 0, 200)  # Rojo
            thic = 2
            if start_times[pos_tuple] is None:
                start_times[pos_tuple] = time.time()
        else:  # Si no hay auto detectado
            color = (0, 200, 0)  # Verde
            thic = 5
            spaces += 1
            if start_times[pos_tuple] is not None:
                start_times[pos_tuple] = None
                occupied_times[pos_tuple] = 0  # Reiniciar el tiempo de ocupación

        # Calcular y mostrar el tiempo de ocupación en horas, minutos y segundos
        occupied_time = occupied_times[pos_tuple]
        if start_times[pos_tuple] is not None:
            occupied_time += time.time() - start_times[pos_tuple]

        hours, rem = divmod(occupied_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_text = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'

        # Dibujar el polígono del espacio y mostrar el tiempo de ocupación
        cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=thic)
        cv2.putText(img, time_text, (polygon[0][0][0], polygon[0][0][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    # Mostrar el estado de los espacios libres/ocupados
    cvzone.putTextRect(img, f'Free: {spaces}/{len(posList)}', (30, 40), thickness=3, offset=20, colorR=(0, 200, 0))

    return spaces, len(posList) - spaces

# Función para procesar los fotogramas de la cámara y mostrar los espacios ocupados/libres
def process_frame():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    while True:
        success, img = cap.read()  # Capturar la imagen de la cámara
        if not success:
            print("Error al capturar el fotograma.")
            break

        # Revisar los espacios de estacionamiento y calcular los tiempos de ocupación
        free_spaces, occupied_spaces = checkSpaces(img)

        # Guardar el estado de los espacios (libres/ocupados) en un archivo .pkl
        with open(spaces_status_filename, 'wb') as f:
            pickle.dump((free_spaces, occupied_spaces), f)

        # Mostrar la imagen con los espacios de estacionamiento marcados
        cv2.imshow("Image", img)

        # Presionar 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frame()
