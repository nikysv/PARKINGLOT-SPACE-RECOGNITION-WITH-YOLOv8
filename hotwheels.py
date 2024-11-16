import cv2
from ultralytics import YOLO

# Cargar el modelo entrenado
from ultralytics import YOLO
model = YOLO('yolov8m.pt')  # Puedes usar la versión pequeña (yolov8n) o una más grande



# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(1)  # 0 es el índice de la cámara predeterminada, ajusta si tienes varias cámaras

# Verificar si la cámara está abierta correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    # Capturar cuadro por cuadro
    ret, frame = cap.read()

    if not ret:
        print("No se pudo recibir el cuadro. Saliendo...")
        break

    # Realizar inferencia en el cuadro capturado
    results = model(frame)

    # Obtener el cuadro anotado con los bounding boxes
    annotated_frame = results[0].plot()

    # Mostrar el cuadro con las detecciones
    cv2.imshow('Detección en tiempo real', annotated_frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
