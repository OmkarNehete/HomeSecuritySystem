import cv2
import numpy as np

cam = cv2.VideoCapture(0)
faceCascade=cv2.CascadeClassifier('C:\\Users\\omkar\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

Id=input('enter your id')
name=input('enter your name')
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the 1
        #captured face in the dataset folder
        cv2.imwrite("ImageDataSet/dataset"+Id +name+'.'+ str(sampleNum) +".jpg", gray)
        cv2.imshow('frame',img)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>100:
        break
cam.release()
cv2.destroyAllWindows()


