'''######################################
This is the implementation of CNN using Keras
'''######################################

import keras
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.preprocessing import image
from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from time import time   
from PIL import Image
'''######################################
Configuration to use cpu only
'''######################################
# config = tf.ConfigProto()
config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 4} ) 
session = tf.Session(config=config)
keras.backend.set_session(session)

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss =[] # loss at given epoch

    def on_epoch_begin(self,epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        #add validation accuracy and loss
        self.model.save('./models-epoch/jsteg-%d.h5' %epoch)
        
        # save images every epoch
        plt.plot(self.acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig("plots/jsteg-%d.png" %epoch) # saves the rate as an image


class SteganalysisModel:
    def __init__(self):
        self.IMAGE_SIZE = 512        
        self.PATH = os.getcwd()
        self.BS = 100 # batch size
        self.createModel()
        self.trainModel()
        self.saveModel()
    

    '''################################
    Creating Convolutional Model:
    Input -> ConvLayer -> 3x3x3 kernel-> TanH -> ConvLayer -> 509x509x64 -> TanH -> fully connected -> ReLU -> Softmax ->output
    '''################################    
    def createModel(self):

        self.model = Sequential()

        # change to 
        # input_shape=(self.IMAGE_SIZE,self.IMAGE_SIZE,1)
        # if the input is grayscale
        self.model.add(Conv2D(1,(3,3), input_shape=(self.IMAGE_SIZE,self.IMAGE_SIZE,3)))
        self.model.add(Activation('tanh'))

        self.model.add(Conv2D(64,(509,509)))
        self.model.add(Activation('tanh'))

        self.model.add(Flatten())

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))

    '''################################
    # This is where the training happens.
    You can lower the 'epochs' in model.fit to lessen the processing time
    '''################################ 
    def trainModel(self):
        
        X_train,y_train = self.getTrainingData()
        X_test,y_test = self.getTestingData()
        


        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        sgd = keras.optimizers.SGD(lr=0.005,decay=5e-7,momentum=0.0)

        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'], options = run_opts)
        self.history = AccuracyHistory()


        self.model.fit(
            X_train, 
            y_train, 
            batch_size=100, 
            validation_data=(X_test, y_test), 
            epochs=45, callbacks=[self.history], shuffle=True)
        self.model.summary()
        score = self.model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        plt.plot(self.history.acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig("jsteg.png") # saves the rate as an image
        plt.show()

    '''################################
    loads all the training data
    '''################################ 
    def getTrainingData(self):
        COVER = 0
        STEGO = 1
        # [COVER, STEGO]
        cover_path = self.PATH+'/main_dataset/train/orig'
        stego_path = self.PATH+'/main_dataset/train/jsteg'

        cover_images = os.listdir(cover_path) #lists all the files in the folder
        stego_images = os.listdir(stego_path) #lists all the files in the folder

        X_train = []
        y_train = []
        train = {}
        for img in cover_images:
            try:
                img_path = cover_path+'/'+img
                # change to 
                # x = Image.open(img_path)
                # if the input is grayscale
                x = image.load_img(img_path).convert('RGB')
                X_train.append(np.array(x))
                y_train.append(COVER)
            except Exception as e:
                print(e)


        for img in stego_images:
            try:
                img_path = stego_path+'/'+img
                # change to 
                # x = Image.open(img_path)
                # if the input is grayscale
                x = image.load_img(img_path).convert('RGB')
                X_train.append(np.array(x))
                y_train.append(STEGO)
            except Exception as e:
                print(e) 

        y_train = np.array(y_train)
        y_train = to_categorical(y_train)
        X_train = np.array(X_train)
        # change to 
        # X_train = X_train.reshape(-1,self.IMAGE_SIZE,self.IMAGE_SIZE,1)
        # if the input is grayscale
        X_train = X_train.reshape(-1,self.IMAGE_SIZE,self.IMAGE_SIZE,3)

        return(X_train,y_train)

    '''################################
    loads all the testing data
    '''################################
    def getTestingData(self):
        COVER = 0
        STEGO = 1
        
        cover_path = self.PATH+'/main_dataset/test/orig'
        stego_path = self.PATH+'/main_dataset/test/jsteg'

        cover_images = os.listdir(cover_path) #lists all the files in the folder
        stego_images = os.listdir(stego_path) #lists all the files in the folder

        X_test = []
        y_test = []
        test = {}
        for img in cover_images:
            try:
                img_path = cover_path+'/'+img
                # change to 
                # x = Image.open(img_path)
                # if the input is grayscale
                x = image.load_img(img_path).convert('RGB')
                X_test.append(np.array(x))
                y_test.append(COVER)
            except Exception as e:
                print(e)


        for img in stego_images:
            try:
                img_path = stego_path+'/'+img
                # change to 
                # x = Image.open(img_path)
                # if the input is grayscale
                x = image.load_img(img_path).convert('RGB')
                X_test.append(np.array(x))
                y_test.append(STEGO)
            except Exception as e:
                print(e) 

        y_test = np.array(y_test)
        y_test = to_categorical(y_test)
        X_test = np.array(X_test)
        # change to 
        # X_train = X_train.reshape(-1,self.IMAGE_SIZE,self.IMAGE_SIZE,1)
        # if the input is grayscale
        X_test = X_test.reshape(-1,self.IMAGE_SIZE,self.IMAGE_SIZE,3)
        return(X_test,y_test)
        

    '''################################
    Saves the whole neural network with all the parameters, weights and biases.
    '''################################ 
    def saveModel(self):
        self.model.save('./models/hugo.h5')

    '''################################
    classifies the image given
    '''################################
    def predictImage(self, imgPath):
        img = Image.open(imgPath)
        img = np.array(img)
        img = img.reshape(1, self.IMAGE_SIZE, self.IMAGE_SIZE,3)
        prediction = self.model.predict(img)
        print("predicting: ",prediction)

def main():
    #create class
    network = SteganalysisModel()


if __name__ == "__main__":
    main()