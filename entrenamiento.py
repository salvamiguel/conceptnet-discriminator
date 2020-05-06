import json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization as BN
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GaussianNoise as GN
from keras.optimizers import *
from keras.callbacks import *
import argparse

def entrenar(archivo_procesado, epochs=40):
    r = open(archivo_procesado, "r")
    lineas = r.readlines()
    palabras = []
    for linea in lineas:
        palabras.append(json.loads(linea))

    palabras = sorted(palabras)
    
    for palabra in palabras:
        palabra.pop(0)

    x = []
    y = []

    for linea in palabras:
        y.append(linea[5])
        if len(linea[3][0]) is not 39 or len(linea[3][1]) is not 39:
            print(linea)
            print(len(linea[3][0]))
            print(len(linea[3][1]))
            return
        salida_1 = np.array(linea[3][0]).reshape(39,1)
        entrada_1 = np.array(linea[3][1]).reshape(1, 39)
        if len(linea[4][0]) is not 39 or len(linea[4][1]) is not 39:
            print(linea)
            print(len(linea[4][0]))
            print(len(linea[4][1]))
            return
        salida_2 = np.array(linea[4][0]).reshape(39,1)
        entrada_2 = np.array(linea[4][1]).reshape(1, 39)
        #img_salida = np.dot(salida_1, salida_2)
        #img_entrada = np.dot(entrada_1, entrada_2)
        img_1 = np.dot(salida_1, entrada_1)
        img_2 = np.dot(salida_2, entrada_2)
        
        x.append([img_1, img_2])
    x = np.array(x)
    x = x.reshape(len(palabras),39,39,2)

    y = np.array(y)
    #print(x.shape)
    #print(y.shape)
    
    #create model
    model = Sequential()
    #add model layers
    gn = 0.2
    model.add(Conv2D(8, kernel_size=3, activation='relu', padding='valid', strides=(1,1), input_shape=(39,39,2)))
    model.add(BN())
    model.add(GN(gn))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='valid', strides=(1,1)))    
    model.add(BN())
    model.add(GN(gn))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(BN())
    model.add(GN(gn))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #model.add(Dense(288, activation='relu'))
    model.add(Dense(2, activation='relu'))

    model.summary()
    #return
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.75, nesterov=True)
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-7, schedule_decay=0.004)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            mode='min',
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
                          
    history= model.fit(X_train, y_train, 
                            epochs=epochs,
                            validation_data=(X_test, y_test),
                            callbacks=[learning_rate_reduction],
                            verbose=1)
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
    y_pred = model.predict(x)
    #print(y_pred)
    #print(y_pred.shape)
    for threshold in np.arange(0.35,1,0.05):
        y_det = list(map(lambda x: x[0] ^ x[1], np.where(y_pred > threshold, 1, 0)))
    #print(y.tolist())
    #print(y_det.tolist())
        print(f1_score(y.tolist(), y_det))

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--entrada', help='Ruta del archivo de entrada')
#argparser.add_argument('-s', '--salida', help='Nombre de la población')
argparser.add_argument('-e', '--epochs', help='Número de epochs', default=40)
args = argparser.parse_args()

entrenar(args.entrada, args.epochs)




## OPTIM AND COMPILE opt = SGD(lr=0.01, decay=1e-6,momentum=0.75, nesterov=True) 
# #opt = rmsprop(lr=0.001,decay=1e-6) 
# #opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-7, schedule_decay=0.004)  
# model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])  
# 
# # DEFINE A LEARNING RATE SCHEDULER learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',   
# 
#  DEFINE A LEARNING RATE SCHEDULER learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', mode='max', patience=5, verbose=1, factor=0.5,                                              min_lr=0.0001) 19:04                                          mode='max',                                             patience=5, 
## TRAINING with DA and LRA history=model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),                             steps_per_epoch=len(x_train) / batch_size,                             epochs=epochs,                             validation_data=testdatagen.flow(x_test, y_test),                             validation_steps=len(x_train) / batch_size,                             callbacks=[learning_rate_reduction],                             verbose=1)
#import matplotlib.pyplot as plt #  "Accuracy" plt.plot(history.history['acc']) plt.plot(history.history['val_acc']) plt.title('model accuracy') plt.ylabel('accuracy') plt.xlabel('epoch') plt.legend(['train', 'validation'], loc='upper left') plt.show()