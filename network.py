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

'''######################################
Configuration to use cpu only
'''######################################
# config = tf.ConfigProto()
config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 4} ) 
session = tf.Session(config=config)
keras.backend.set_session(session)


class SteganalysisModel:
    def __init__(self):
        self.IMAGE_SIZE = 512        
        self.PATH = os.getcwd()
        self.createModel()
        self.trainModel()
        self.saveModel()
    

    '''################################
    Creating Convolutional Model:
    Input -> ConvLayer -> 3x3x3 kernel-> Relu -> ConvLayer -> 509x509x64 -> Relu -> fully connected -> output
    '''################################    
    def createModel(self):

        self.model = Sequential()

        self.model.add(Conv2D(3,(3,3), input_shape=(self.IMAGE_SIZE,self.IMAGE_SIZE,3)))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(64,(509,509)))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(4, activation='softmax'))

    '''################################
    This is where the training happens.
    You can lower the 'epochs' in model.fit to lessen the processing time
    '''################################ 
    def trainModel(self):
        
        X_train,y_train = self.getTrainingData()
        X_test,y_test = self.getTestingData()
        

        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'], options = run_opts)

        self.model.fit(
            X_train, 
            y_train, 
            batch_size=10, 
            validation_data=(X_test, y_test), 
            epochs=5)

        self.model.summary()
        score = self.model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        

    '''################################
    Saves the whole neural network with all the parameters, weights and biases.
    '''################################ 
    def saveModel(self):
        self.model.save('./models/CNN_weights4.h5')

    '''################################
    loads all the training data
    '''################################ 
    def getTrainingData(self):
        train_path = self.PATH+'/main_dataset/training_data/'
        train_categories = os.listdir(train_path) #lists all the files in the folder

        X_train = []
        y_train = []
        for label in train_categories:
            train_batch = os.listdir(train_path+label)
            # print(label)
            #outguess, f5,orig, jsteg

            for img in train_batch:
                try:
                    img_path = train_path+label+'/'+img
                    x = image.load_img(img_path).convert('RGB')
                    X_train.append(np.array(x))
                    # X_train.append(x)
                    y_train.append(train_categories.index(label))
                except:
                    pass
        y_train = np.array(y_train)
        y_train = to_categorical(y_train)
        X_train = np.array(X_train)
        X_train = X_train.reshape(-1,self.IMAGE_SIZE,self.IMAGE_SIZE,3)

        return(X_train,y_train)

    '''################################
    loads all the testing data
    '''################################
    def getTestingData(self):
        X_test = []
        y_test = []
        train_path2 = self.PATH+'/main_dataset/testing_data/'
        train_categories2 = os.listdir(train_path2) #lists all the files in the folder

        for label in train_categories2:
            train_batch = os.listdir(train_path2+label)

            for img in train_batch:
                try:
                    img_path = train_path2+label+'/'+img
                    x = image.load_img(img_path).convert('RGB')
                    X_test.append(np.array(x))
                    # X_test.append(x)
                    y_test.append(train_categories2.index(label))
                except:
                    pass
                    
        
        y_test = np.array(y_test)
        y_test = to_categorical(y_test)
        X_test = np.array(X_test)
        X_test = X_test.reshape(-1,self.IMAGE_SIZE,self.IMAGE_SIZE,3)

        return(X_test, y_test)

    '''################################
    classifies the image given
    '''################################
    def predictImage(self, imgPath):
        img = image.load_img(imgPath, target_size=(self.IMAGE_SIZE,self.IMAGE_SIZE))
        img = np.array(img)
        img = img.reshape(1, self.IMAGE_SIZE, self.IMAGE_SIZE,3)
        prediction = self.model.predict(img)
        print("predicting: ",prediction)

def main():
    #create class
    network = SteganalysisModel()
    network.predictImage("/home/jcooo/Desktop/Agustin (DO NOT DELETE)/Agustin, Jerico SP/main_dataset/testing_data/f5/2000.jpg")
if __name__ == "__main__":
    main()