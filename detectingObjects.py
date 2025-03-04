import cv2
import numpy as np

classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "library", "bus", "car",
    "church", "cow", "dog", "horse", "cookie", "cake", "person", "sheep", "fireman",
    "sofa", "superman", "train", "tvmonitor", "streets", "laptop", "doctor", "keyboard",
    "nurse", "microwave", "belt", "buildings", "dress", "rug", "bed", "remote", "lamp",
    "bin", "bean", "gloves", "scarf", "charger", "basket", "toy", "door", "gameconsole",
    "painting", "wallet", "diningtable", "candle", "watch", "fan", "speaker", "flowerpot",
    "chairmat", "clock", "mouse", "printer", "hat", "tape", "cup", "book", "corn", "phone"
]

image = cv2.imread("testImg.jpg")
if image is None:
    raise FileNotFoundError("Imaginea 'testImg.jpg' nu a fost găsită. Asigură-te că numele și calea sunt corecte.")

scale_factor = 2.5
(h, w) = image.shape[:2]

num_detections = 8
np.random.seed(42)

for i in range(num_detections):
    confidence = round(np.random.uniform(0.4, 1.0), 2)
    idx = np.random.randint(1, len(classes))

    startX = np.random.randint(0, w // 2)
    startY = np.random.randint(0, h // 2)
    endX = np.random.randint(w // 2, w - 10)
    endY = np.random.randint(h // 2, h - 10)

    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 3)
    label = f"{classes[idx]}: {confidence * 100:.2f}%"
    cv2.putText(image, label, (startX, max(startY - 15, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.namedWindow("Detecție Obiecte (Simulată)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detecție Obiecte (Simulată)", 1000, 800)
cv2.imshow("Detecție Obiecte (Simulată)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()