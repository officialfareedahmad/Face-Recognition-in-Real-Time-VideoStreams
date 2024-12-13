# Face Recognition using TensorFlow and OpenCV

import cv2
import os
import numpy as np
from tensorflow import keras
from keras.models import load_model


facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX

try:
    model = load_model('keras_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

def get_className(classNo):
   if classNo == 0:
      return "Fareed"
   elif classNo == 1:
      return "Salman"


while True:
   success, imgOriginal = cap.read()
   if not success:
      break
   faces = facedetect.detectMultiScale(imgOriginal, 1.3,5)
   for x,y,w,h in faces:
      crop_img = imgOriginal[y:y+h, x:x+w]
      img = cv2.resize(crop_img, (224, 224))
      img = img.reshape(1,224,224,3)
      prediction = model.predict(img)
      classIndex= model.predict_classes(img)
      probabilityValue=np.amax(prediction)
      if classIndex == 0:
         cv2.rectangle(imgOriginal, (x,y), (x+w, y+h) , (0, 225, 0),2)
         cv2.rectangle(imgOriginal,(x,y-40), (x+w, y),(0,255,0),-2)
         cv2.putText(imgOriginal, str(get_className(classIndex)), (x,y-10), font , 0.75, (255,255,255), 2)
      elif classIndex == 1:
         cv2.rectangle(imgOriginal, (x,y), (x+w, y+h) , (0,255,0),2)
         cv2.rectangle(imgOriginal,(x,y-40), (x+w, y),(0,255,0),-2)
         cv2.putText(imgOriginal, str(get_className(classIndex)), (x,y-10), font , 0.75, (255,255,255), 2)

      cv2.putText(imgOriginal, str(round(probabilityValue*100,2)) + "%" , (180,75), font, 0.75, (0,0,255), 2)

   cv2.imshow("Result", imgOriginal)
   k=cv2.waitKey(1)
   if k==ord('q'):
      break
   
cap.release()
cv2.destroyAllWindows()