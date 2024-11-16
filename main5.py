import cv2
import pickle
import cvzone
import numpy as np
import time
import os
import mysql.connector
from datetime import datetime
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8m.pt')

# Obtener el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas para los archivos de posiciones y estado de espacios
car_park_pos_filename = os.path.join(script_dir, 'CarParkPos.pkl')
spaces_status_filename = os.path.join(script_dir, 'spaces_status.pkl')

# Conexión a la base de datos MySQL
db_config = {
    'user': 'root',
    'password': 'mysql',
    'host': 'localhost',
    'database': 'parking_system'
}
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Crear tabla si no existe
cursor.execute('''CREATE TABLE IF NOT EXISTS parking_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    space_number INT NOT NULL,
                    llegada DATETIME NOT NULL,
                    salida DATETIME,
                    fecha DATE NOT NULL,
                    total DECIMAL(10, 2)
                    )''')
conn.commit()

# Cargar la lista de posiciones de los espacios de estacionamiento
try:
    with open(car_park_pos_filename, 'rb') as f:
        posList = pickle.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{car_park_pos_filename}'. Asegúrate de que el archivo exista.")
    exit()
print(posList)

# Diccionario para almacenar el tiempo de ocupación de cada espacio
start_times = {tuple(map(tuple, pos)): None for pos in posList}

# Función para registrar en la base de datos MySQL
def log_parking_time(space_number, start_time, end_time):
    total_minutes = (end_time - start_time).total_seconds() / 60
    total_cost = round(total_minutes * 0.20, 2)  # 0.20 centavos por minuto

    # Registrar en la base de datos
    fecha = start_time.date()
    cursor.execute('''INSERT INTO parking_log (space_number, llegada, salida, fecha, total)
                      VALUES (%s, %s, %s, %s, %s)''',
                   (space_number, start_time, end_time, fecha, total_cost))
    conn.commit()

# Función para verificar los espacios y detectar autos usando YOLO
def checkSpaces(img):
    spaces = 0
    results = model(img, conf=0.25)

    for i, pos in enumerate(posList):
        polygon = np.array(pos, np.int32).reshape((-1, 1, 2))

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

        pos_tuple = tuple(map(tuple, pos))

        if detected_car:  # Si se detecta un auto
            color = (0, 0, 200)  # Rojo
            if start_times[pos_tuple] is None:
                # Registrar la hora de llegada
                start_times[pos_tuple] = time.time()  # Guardar el tiempo como timestamp
        else:  # Si no hay auto detectado
            color = (0, 200, 0)  # Verde
            spaces += 1
            if start_times[pos_tuple] is not None:
                # Registrar la salida y el tiempo total en la base de datos
                end_time = time.time()
                log_parking_time(i + 1, datetime.fromtimestamp(start_times[pos_tuple]), datetime.fromtimestamp(end_time))
                start_times[pos_tuple] = None  # Reiniciar el espacio

        # Calcular y mostrar el tiempo de ocupación en horas, minutos y segundos si está ocupado
        if start_times[pos_tuple] is not None:
            occupied_time = time.time() - start_times[pos_tuple]
            hours, rem = divmod(occupied_time, 3600)
            minutes, seconds = divmod(rem, 60)
            time_text = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'

            # Mostrar el tiempo de ocupación sobre el espacio en la imagen
            cv2.putText(img, time_text, (polygon[0][0][0], polygon[0][0][1] - 10), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        # Dibujar el polígono del espacio
        cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=1)

        # Mostrar el número del espacio en la esquina inferior derecha del polígono
        bottom_right_x = polygon[2][0][0]  
        bottom_right_y = polygon[2][0][1]
        cv2.putText(img, f'{i + 1}', (bottom_right_x - 40, bottom_right_y - 10), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Mostrar el estado de los espacios libres/ocupados
    cvzone.putTextRect(img, f'Free: {spaces}/{len(posList)}', (30, 40), thickness=3, offset=20, colorR=(0, 200, 0))

    return spaces, len(posList) - spaces

# Función para procesar los fotogramas de la cámara y mostrar los espacios ocupados/libres
def process_frame():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    while True:
        success, img = cap.read()
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_frame()

    # Cerrar la conexión con la base de datos al final
    conn.close()
