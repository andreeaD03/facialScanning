import cv2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
imagine = cv2.imread("image22.jpg")
grayImage = cv2.cvtColor(imagine, cv2.COLOR_BGR2GRAY)

#detectam fetele
faces = faceCascade.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

 #desenam un dreptunghi in jurul fiecarei fete detectate
for(x,y,w,h) in faces:
    cv2.rectangle(imagine, (x,y), (x+w, y+h), (0,255,0), 2)


cv2.imshow("detectare faciala", imagine)

cv2.waitKey(0)
cv2.destroyAllWindows()
