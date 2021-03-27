import cv2
import numpy as np

vid= cv2.VideoCapture(0)#Captures the webcam (0) represents your inbuild webcam for external camera (1) 
faceCascade = cv2.CascadeClassifier('C:\\Users\\Denni\\Desktop\\haarcascade_frontalface_default.xml')#location of xml file initialize with \\

while True:
    ret,video = vid.read()
    gray = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)#Makes RGB to GRAY for making the classifier to find the face easier 
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(video, (x, y), (x+w, y+h), (0,0,255), 2)#Draw's a rectangle around the detection area
    img2 = np.array(video)
    cv2.imshow("video", img2)#Shows the output as window
    if cv2.waitKey(1) & 0xFF == ord('q'):#Make (Q) as quit option
        break

    
vid.release()
cv2.destroyAllWindows()#Breaks all the window and releases camera
