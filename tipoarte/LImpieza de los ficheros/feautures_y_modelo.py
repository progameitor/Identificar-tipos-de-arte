import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import glob
import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

carpeta = ["cubism_128/","expressionism_128/","fauvism_128_espejo/","pointillism_126_espejo/","pop_art_128_espejo/",'ukiyo_128']

def obtener_Ruta(carpeta):
    first = 'Inputs/imagenes_extraidas/'
    last = '*.jpg'
    result = first + carpeta + last
    return result

obtener_Ruta('expressionism_128/')

diccionario = {'cubism_128/': 1,
               'expressionism_128/':2,
               "fauvism_128_espejo/":3,
               "pointillism_126_espejo/":4,
               "pop_art_128_espejo/":5,
               'ukiyo_128/':6            
               }

def sacar_Imagenes(carpeta):
    lista = []
    for x in glob.glob(obtener_Ruta(carpeta)):
        for k,v in diccionario.items():
            if k in x:
                lista.append(v)
    return lista

z = []

for x in diccionario:
    z.append(sacar_Imagenes(x))

for y in z:
    np.asarray(y)
y

array = []
for x in z:
    array.append(np.asarray(x, dtype = np.int64))
array

# estaria guay crearme una funcion para no repetir todo el rato

concat1 =array[0]
concat2 = array[1]
concat3 = array[2]
concat4 = array[3]
concat5 = array[4]
concat6 = array[5]

# aqui he obtenido ya mi grant truth que le paso el modelo
array_y= np.concatenate((concat1, concat2,concat3,concat4,concat5,concat6), axis=0)
array_y

array_y.shape

# Aqui recorro todas mis rutas y saco los ficheros que me interesan
im = [] 
for file in glob.glob('Inputs/imagenes_extraidas/*/*.jpg'):
    im.append(cv2.imread(file)/255)
    
# ya tengo mi ruta    
X =np.asarray(im)
X.shape

#preparo el modelo para entrenarlo
X_train, X_test, y_train, y_test = train_test_split(X, array_y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#numero de clases y la forma de mi matriz
num_classes = 6
input_shape = (128, 128, 3)

#NN topology

model = Sequential()
inputShape = (128, 128, 3)
chanDim = -1
if K.image_data_format() == "channels_first":
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation("softmax"))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# Esto sirve para guardar los mejores modelos
filepath='Checkpoint_{epoch:02d}_{val_acc:.2f}'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# entrenamiento del modelo
batch_size = 125
epochs = 100

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=callbacks_list)

