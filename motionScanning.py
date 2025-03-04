
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

classes = [
    "background", "bottle", "bus", "car", "cat", "chair", "dog", "horse", "person", "sofa", "train", "tvmonitor", "hand", "face", "eyes", "hair", "phone"
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Nu s-a putut accesa camera web.")

np.random.seed(42)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame_resized, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"Object: {np.random.choice(classes[1:])}"
            cv2.putText(frame_resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Detecție față și obiecte (simulată)', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()