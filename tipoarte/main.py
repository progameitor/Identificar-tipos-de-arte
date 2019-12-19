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
import keras as tf
import pandas as pd
import time
import cv2
import json
import time
from os import listdir
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential,load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D,Conv2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, ImageDataGenerator, image
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from Src.funciones import *

directorio =os.listdir()
directorio

carpeta = ["cubism_128/","expressionism_128/","fauvism_128_espejo/","pointillism_126_espejo/","pop_art_128_espejo/",'ukiyo_128']

print(obtener_Ruta('cubism_128/'))

diccionario = {'cubism_128/': 1,
               'expressionism_128/':2,
               "fauvism_128_espejo/":3,
               "pointillism_126_espejo/":4,
               "pop_art_128_espejo/":5,
               'ukiyo_128/':6            
               }

print(sacar_Imagenes('cubism_128/'))

z = []

for x in diccionario:
    z.append(sacar_Imagenes(x))
print(z)

array = []
for x in z:
    array.append(np.asarray(x, dtype = np.int64))
array

# concateno mis arrays para que puedan entran 
y = np.concatenate((array[0],array[1],array[2],array[3],array[4],array[5]),axis = 0)
print(len(y))
print(y.shape)

# paso las variables a categoricas para que me lo reconozca keras
y = tf.utils.to_categorical(
    y,
    num_classes=6,
    dtype='float32'
)
y.shape

# aqui me importo las imagenes para normarlizarlas y que pueda cogerlas la red neuronal. Esto es mi X
im = [] 
for file in glob.glob('Inputs/imagenes_extraidas/128x128/*/*.jpg'):
    im.append(cv2.imread(file)/255)

X =np.asarray(im)
print(X.shape)


