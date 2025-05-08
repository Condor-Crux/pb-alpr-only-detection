import cv2
import pytesseract
from yolov5 import YOLOv5  # si usás el wrapper yolov5, o torch.hub si es manual

model = YOLOv5("best.pt", device="cpu")  # o "cuda" si tenés GPU
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # o ruta donde está tesseract

cap = cv2.VideoCapture("rtsp://localhost:8554/mystream")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        roi = frame[y1:y2, x1:x2]
        text = pytesseract.image_to_string(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), config='--psm 7')
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text.strip(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Detección de Patentes", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
