import pdb
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ap = argparse.ArgumentParser()
# ap.add_argument("--filename",help="sad.jpg")
# filename = ap.parse_args().filename


class EmotionDetector():
    def __init__(self, weights_path):
        # Create the model
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))


        # model.load_weights('emotion_detection/model.h5')
        model.load_weights(weights_path)
        self.model = model

        # dictionary which assigns each label an emotion (alphabetical order)
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


    def predict(self, img_array):
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
        # frame = cv2.imread(filename)
        frame = img_array
        # print(img_array.shape)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(gray.shape)

        emotions = []
        confidence = []
        # try:
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5)       
            
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = self.model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            confidence.append(float(prediction[0][maxindex])*100)
            emotions.append(self.emotion_dict[maxindex])
            # cv2.putText(frame, self.emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # except:
        #     emotions.append('No emotions detected!')
            
        return emotions, confidence
# # cv2.imwrite('test_'+filename, cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
# cv2.imwrite(f'test_{filename.split("/")[-1]}', frame)
