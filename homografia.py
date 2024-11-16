import cv2
import numpy as np
import pickle  # Cambiado de json a pickle

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
            cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
            parking_spaces.append(points.copy())  # Guardar el espacio en la lista
            points.clear()  # Limpiar los puntos después de dibujar el área
            print(f"Espacio definido: {parking_spaces[-1]}")

# Para guardar las coordenadas de los espacios en un archivo pickle
def save_parking_spaces(spaces, filename='parking_spaces.pkl'):  # Cambiado de .json a .pkl
    with open(filename, 'wb') as f:  # wb para escribir en binario
        pickle.dump(spaces, f)
    print(f"Espacios guardados en {filename}")

# Capturar imagen desde la cámara
#cap = cv2.VideoCapture(0)
# 0 es el índice de la cámara; ajusta si tienes múltiples cámaras
cap = cv2.imread()

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
            save_parking_spaces(parking_spaces)  # Guardar los espacios definidos en pickle
            break

    cap.release()
    cv2.destroyAllWindows()
