import cv2
import numpy as np
import pickle
import os

# Obtener el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Lista para almacenar los puntos seleccionados por el usuario
points = []
parking_spaces = []  # Lista para almacenar las coordenadas de los espacios definidos

# Variable para controlar si se ha capturado la imagen
image_captured = False

# Función de callback para manejar los clics del usuario
def select_points(event, x, y, flags, param):
    global points, image, parking_spaces
    if event == cv2.EVENT_LBUTTONDOWN and image_captured:
        points.append((x, y))  # Guardar las coordenadas de los puntos
        if len(points) == 4:  # Cuando se seleccionen 4 puntos
            # Dibujar el polígono con los puntos seleccionados sin modificar su ángulo
            cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=1)
            
            parking_spaces.append(points.copy())  # Guardar el espacio en la lista
            points.clear()  # Limpiar los puntos después de dibujar el área
            print(f"Espacio definido: {parking_spaces[-1]}")

# Para guardar las coordenadas de los espacios en un solo archivo
def save_parking_spaces(spaces, filename='CarParkPos.pkl'):
    filepath = os.path.join(script_dir, filename)
    with open(filepath, 'wb') as f:  # Guardar en un solo archivo
        pickle.dump(spaces, f)
    print(f"Espacios guardados en {filepath}")

# Para guardar el estado de los espacios (libre/ocupado) en un archivo .pkl
def save_spaces_status(status, filename='spaces_status.pkl'):
    filepath = os.path.join(script_dir, filename)
    with open(filepath, 'wb') as f:  # Guardar en binario
        pickle.dump(status, f)
    print(f"Estado de los espacios guardado en {filepath}")

# Para guardar la imagen capturada
def save_image(image, filename='parking_image.jpg'):
    filepath = os.path.join(script_dir, filename)
    cv2.imwrite(filepath, image)  # Guardar la imagen en formato JPEG
    print(f"Imagen guardada como {filepath}")

# Capturar imagen desde la cámara
cap = cv2.VideoCapture(1)  # 0 es el índice de la cámara; ajusta si tienes múltiples cámaras

print("Presiona 'c' para capturar la imagen del estacionamiento.")

while True:
    success, frame = cap.read()
    if not success:
        print("No se pudo acceder a la cámara.")
        break

    cv2.imshow('Captura de Estacionamiento', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        image = frame.copy()
        image_captured = True
        save_image(image)  # Guardar la imagen capturada
        cv2.destroyWindow('Captura de Estacionamiento')
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

if image_captured:
    # Mostrar la imagen capturada y permitir la selección de espacios
    cv2.imshow('Seleccionar espacios de estacionamiento', image)
    cv2.setMouseCallback('Seleccionar espacios de estacionamiento', select_points)

    while True:
        cv2.imshow('Seleccionar espacios de estacionamiento', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            save_parking_spaces(parking_spaces)  # Guardar todos los espacios en un solo archivo

            # Guardar el estado de los espacios (libres/ocupados) en el archivo .pkl
            # Aquí guardamos el estado inicial como todos libres (True = libre)
            initial_status = {tuple(space): True for space in parking_spaces}
            save_spaces_status(initial_status)

            break

    cap.release()
    cv2.destroyAllWindows()
