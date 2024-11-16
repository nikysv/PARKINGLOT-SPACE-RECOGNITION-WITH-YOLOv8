import cv2
import pickle
import cvzone
import numpy as np
import time
import os
import mysql.connector
from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO('yolov8m.pt')  # Puedes cambiar a yolov8m.pt o yolov8l.pt para modelos más grandes

# Obtener el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Rutas para los archivos de posiciones y estado de espacios
car_park_pos_filename = os.path.join(script_dir, 'CarParkPos.pkl')  # Archivo con coordenadas de los espacios
spaces_status_filename = os.path.join(script_dir, 'spaces_status.pkl')  # Archivo con el estado de los espacios

# Conexión a la base de datos MySQL
db_config = {
    'user': 'root',
    'password': 'mysql',
    'host': 'localhost',
    'database': 'parking_system'
}
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Crear tabla si no existe (solo si no la has creado previamente)
cursor.execute('''CREATE TABLE IF NOT EXISTS parking_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    space_number INT NOT NULL,
                    occupied_time DECIMAL(10, 2) NOT NULL,
                    timestamp DATETIME NOT NULL
                    )''')
conn.commit()

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

# Función para registrar en la base de datos MySQL
def log_parking_time(space_number, occupied_time):
    if occupied_time > 10:  # Solo registrar si el tiempo es mayor a 10 segundos
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("INSERT INTO parking_log (space_number, occupied_time, timestamp) VALUES (%s, %s, %s)",
                       (space_number, occupied_time, timestamp))
        conn.commit()

# Función para verificar los espacios y detectar autos usando YOLO
def checkSpaces(img):
    spaces = 0
    results = model(img, conf=0.25)  # Ajusta el umbral de confianza

    for i, pos in enumerate(posList):
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
            thic = 1  # Líneas más delgadas para espacios ocupados
            if start_times[pos_tuple] is None:
                start_times[pos_tuple] = time.time()
        else:  # Si no hay auto detectado
            color = (0, 200, 0)  # Verde
            thic = 1  # Líneas más delgadas para espacios libres
            spaces += 1
            if start_times[pos_tuple] is not None:
                # Calcular el tiempo de ocupación y guardarlo en la base de datos si supera 10 segundos
                occupied_time = time.time() - start_times[pos_tuple]
                log_parking_time(i + 1, occupied_time)  # Registrar el espacio en la base de datos
                start_times[pos_tuple] = None
                occupied_times[pos_tuple] = 0  # Reiniciar el tiempo de ocupación

        # Calcular y mostrar el tiempo de ocupación en horas, minutos y segundos
        occupied_time = occupied_times[pos_tuple]
        if start_times[pos_tuple] is not None:
            occupied_time += time.time() - start_times[pos_tuple]

        hours, rem = divmod(occupied_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_text = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'

        # Dibujar el polígono del espacio, mostrar el tiempo de ocupación y el número de espacio
        cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=thic)  # Líneas más delgadas
        cv2.putText(img, time_text, (polygon[0][0][0], polygon[0][0][1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        # Ajustar la posición del número del espacio en la esquina inferior derecha del polígono
        bottom_right_x = polygon[2][0][0]  # Coordenada x del vértice inferior derecho
        bottom_right_y = polygon[2][0][1]  # Coordenada y del vértice inferior derecho
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

    # Cerrar la conexión con la base de datos al final
    conn.close()
