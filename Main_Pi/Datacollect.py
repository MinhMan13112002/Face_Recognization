# pip install opencv-python==4.5.2

import cv2 
import os
from picamera2 import Picamera2

video=Picamera2()
video.preview_configuration.main.size = (640, 360)
video.preview_configuration.main.format = "RGB888"
video.preview_configuration.controls.FrameRate=30
video.preview_configuration.align()
video.configure("preview")
video.start()
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Enter Your ID: ")
# id = int(id)
count=0

while True:
    frame=video.capture_array()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        count=count+1
        cv2.imwrite('dataset/User.'+str(id)+"."+str(count)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    cv2.imshow("Frame",frame)

    k=cv2.waitKey(100)

    if count>20:
        break

video.stop()
cv2.destroyAllWindows()
print("Dataset Collection Done..................")
