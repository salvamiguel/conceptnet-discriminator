
import os
import argparse
import json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization as BN
from keras.optimizers import *
from keras.callbacks import *
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
import conceptnet


def read_digested(digested_file_path):
    r = open(digested_file_path, "r")
    lines = r.readlines()
    triplets = []
    for line in lines:
        triplets.append(json.loads(line))

    triplets = sorted(triplets)
    
    for line in triplets:
        line.pop(0)
    
    return triplets

    
def train(digested_file_path, epochs=50, opt_type=0):

       
    triplets = read_digested(digested_file_path)
    
    x = []
    y = []
    for line in triplets:
        y.append(line[5])

        print("Reading " + str(line[0]) + ", " + str(line[1]) + ", " + str(line[2]))

        if len(line[3][0]) is not 50 or len(line[3][1]) is not 50:
            print(line)
            print(len(line[3][0]))
            print(len(line[3][1]))
            return

        salida_1 = np.array(line[3][0]).reshape(50,1)
        entrada_1 = np.array(line[3][1]).reshape(1, 50)

        if len(line[4][0]) is not 50 or len(line[4][1]) is not 50:
            print(line)
            print(len(line[4][0]))
            print(len(line[4][1]))
            return

        salida_2 = np.array(line[4][0]).reshape(50,1)
        entrada_2 = np.array(line[4][1]).reshape(1, 50)

        salida_1 = salida_1 + np.ones(salida_1.shape)
        salida_2 = salida_2 + np.ones(salida_2.shape)

        entrada_1 = entrada_1 + np.ones(entrada_1.shape)
        entrada_2 = entrada_2 + np.ones(entrada_2.shape)

        try:
            similarity_1 = conceptnet.m_embedding.similarity(line[0], line[2])
        except:
            similarity_1 = 0.2
        try:
            similarity_2 = conceptnet.m_embedding.similarity(line[1], line[2])
        except:
            similarity_1 = 0.2

        img_1 = np.dot(salida_1, entrada_1) * similarity_1
        img_2 = np.dot(salida_2, entrada_2) * similarity_2
        #max_ = np.amax([img_1, img_2])
        #img_1 /= max_
        #img_2 /= max_


        x.append([img_1, img_2])
    x = np.array(x)
    x = x.reshape(len(triplets),50,50,2)

    y = np.array(y)
    #print(x.shape)
    #print(y.shape)
    
    #create model
    model = Sequential()
    #add model layers
    model.add(Conv2D(8, kernel_size=3, activation='relu', padding='valid', strides=(1,1), input_shape=(50,50,2)))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='valid', strides=(1,1)))    
    model.add(BN())
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='valid', strides=(1,1)))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='valid', strides=(1, 1)))
    model.add(BN())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='valid', strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    
    opts = [SGD(lr=0.01, decay=1e-6, momentum=0.75, nesterov=True),  Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, schedule_decay=0.004), rmsprop(lr=0.001,decay=1e-6), Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07), Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0)]
    opt = opts[opt_type]
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            mode='min',
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)
               
    history= model.fit(X_train, y_train, 
                            epochs=int(epochs),
                            validation_data=(X_test, y_test),
                            callbacks=[learning_rate_reduction],
                            verbose=1)
    
    model.save(os.path.join(".", "pre-trained/en_conceptnet.h5"))
    model.save_weights(os.path.join(".", 'pre-trained/en_weights.hdf5'), overwrite=True)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation', 'loss', 'val_loss'], loc='upper left')
    plt.show()
    predict = model.predict(X_test)
    report = classification_report(np.argmax(y_test,axis=1), np.argmax(predict, axis=1))
    print(report)

    #model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)



argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input', help='Path to digested file. // Ruta del archivo de entrada.')
argparser.add_argument('-e', '--epochs', help='Number of epochs. // Número de epochs', default=50)
argparser.add_argument('-o', '--opt', help='Optimizer. // Optimizador', default=3)

args = argparser.parse_args()

train(args.input, args.epochs)



# load weights
#print("Preload weights...")
## load weights
#filepath="./checkpoints/"+params['FINAL_WEIGHTS_PATH']
#exists = os.path.isfile(filepath)
#if exists:
#    model.load_weights(filepath)



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