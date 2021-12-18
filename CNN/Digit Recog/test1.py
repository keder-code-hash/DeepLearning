import cv2
from tensorflow import keras
from keras import models
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

def pred():
    testingmodel=load_model('digit_recog.h5')
    l=[1]
    for name in l:
        img=cv2.imread(f'{name}.png')[:,:,0]
        img=np.invert(np.array(img))
        plt.imshow(img)
        plt.show() 

        img=img.reshape(28,28,1)

        # img=img.astype('float32')
        # img/=255
        # img=keras.utils.to_categorical(img,num_classes=1)
        # img=tensorflow.constant(img,shape=[28,28,1])

        prediction=testingmodel.predict(img)
        # print("Value: ",np.argmax(prediction))
        print(prediction)
pred()