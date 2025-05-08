import cv2
import pytesseract
from ultralytics import YOLO

# Ruta a tesseract si no está en el PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Cargar modelo YOLO preentrenado para detección general
model = YOLO("yolov8n.pt")  # puedes usar yolov8s.pt o entrenar uno personalizado para patentes

# Conectarse al stream RTSP
cap = cv2.VideoCapture("rtsp://localhost:8554/mystream")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Detección YOLO
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Podés filtrar por clases si tenés un modelo entrenado para 'license plate'
            # Aquí mostramos todos los objetos detectados
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[y1:y2, x1:x2]

            # OCR con pytesseract
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--psm 7')  # Solo una línea de texto

            # Dibujar resultados
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text.strip(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Patentes Detectadas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
