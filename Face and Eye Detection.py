#!python3
import cv2
import numpy as np
face_detect=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_detect=cv2.CascadeClassifier('haarcascade_eye.xml')
vid=cv2.VideoCapture(0)
while True:
    ret,img=vid.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray=gray[y:y+h,x:x+h]
        roi_color=img[y:y+h,x:x+h]
        eyes=eye_detect.detectMultiScale(roi_gray,1.3,5)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    k=cv2.waitKey(100) & 0x0ff
    if k==27:
        break
vid.release()
cv2.destroyAllWindows()

    
