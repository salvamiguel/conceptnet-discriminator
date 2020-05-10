import json
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.layers import GaussianNoise as GN, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization as BN
from keras.optimizers import *
from keras.callbacks import *
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import brown
import gensim 
from gensim.models import Word2Vec, KeyedVectors
import argparse
import os

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}):
        val_predict = list(map(lambda x: sum(x), np.where(self.model.predict(self.validation_data[0]) > 0.5, 1, 0)))
        val_targ = list(map(lambda x: 1 - x[0], self.validation_data[1]))
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return

def train(digested_file_path, epochs=50, path_embeddings = 'pre-trained/GoogleNews-vectors-negative300.bin.gz'):
    r = open(digested_file_path, "r")
    lineas = r.readlines()
    palabras = []
    for linea in lineas:
        palabras.append(json.loads(linea))

    palabras = sorted(palabras)
    
    corpus = []

    for palabra in palabras:
        palabra.pop(0)
        corpus.append(palabra[0])
        corpus.append(palabra[1])
        corpus.append(palabra[2])
       
    

    print("Loading wordembeddings model")
    if os.path.isfile(path_embeddings):
        m_embedding = KeyedVectors.load_word2vec_format(path_embeddings, binary=True)
    else:
        m_embedding = Word2Vec(brown.sents())
        m_embedding.save('pre-trained/' + digested_file_path.replace("/", "_") + ".embedding")
    

    x = []
    y = []
    for linea in palabras:
        y.append(linea[5])

        print("Reading " + str(linea[0]) + ", " + str(linea[1]) + ", " + str(linea[2]))

        if len(linea[3][0]) is not 50 or len(linea[3][1]) is not 50:
            print(linea)
            print(len(linea[3][0]))
            print(len(linea[3][1]))
            return

        salida_1 = np.array(linea[3][0]).reshape(50,1)
        entrada_1 = np.array(linea[3][1]).reshape(1, 50)

        if len(linea[4][0]) is not 50 or len(linea[4][1]) is not 50:
            print(linea)
            print(len(linea[4][0]))
            print(len(linea[4][1]))
            return

        salida_2 = np.array(linea[4][0]).reshape(50,1)
        entrada_2 = np.array(linea[4][1]).reshape(1, 50)

        salida_1 = salida_1 + np.ones(salida_1.shape)
        salida_2 = salida_2 + np.ones(salida_2.shape)

        entrada_1 = entrada_1 + np.ones(entrada_1.shape)
        entrada_2 = entrada_2 + np.ones(entrada_2.shape)

        try:
            similarity_1 = m_embedding.similarity(linea[0], linea[2])
        except:
            similarity_1 = 0.2
        try:
            similarity_2 = m_embedding.similarity(linea[1], linea[2])
        except:
            similarity_1 = 0.2
    
        img_1 = np.dot(salida_1, entrada_1) * similarity_1
        img_2 = np.dot(salida_2, entrada_2) * similarity_2
        #max_ = np.amax([img_1, img_2])
        #img_1 /= max_
        #img_2 /= max_


        x.append([img_1, img_2])
    x = np.array(x)
    x = x.reshape(len(palabras),50,50,2)

    y = np.array(y)
    #print(x.shape)
    #print(y.shape)
    
    #create model
    model = Sequential()
    #add model layers
    gn = 0
    model.add(Conv2D(16, kernel_size=3, activation='relu', padding='valid', strides=(1,1), input_shape=(50,50,2)))
    model.add(BN())
    model.add(GN(gn))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='valid', strides=(1,1)))    
    model.add(BN())
    model.add(GN(gn))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='valid', strides=(1,1)))
    #model.add(BN())
    #model.add(GN(gn))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(128, kernel_size=3, activation='relu', padding='valid', strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.summary()
    
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.75, nesterov=True)
    #opt = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, schedule_decay=0.004)
    #opt = rmsprop(lr=0.001,decay=1e-6)
    opt = Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07)
    #opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0)
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
    metrics = Metrics()
               
    history= model.fit(X_train, y_train, 
                            epochs=int(epochs),
                            validation_data=(X_test, y_test),
                            callbacks=[metrics],
                            verbose=1)
    
    model.save(os.path.join(".", "pre-trained/Conceptnet.h5"))
    model.save_weights(os.path.join(".", 'pre-trained/FINAL_WEIGHTS.hdf5'), overwrite=True)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation', 'loss', 'val_loss'], loc='upper left')
    plt.show()
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)



argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input', help='Path to digested file. // Ruta del archivo de entrada.')
argparser.add_argument('-e', '--epochs', help='Number of epochs. // Número de epochs', default=50)
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