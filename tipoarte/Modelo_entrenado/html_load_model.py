# defino las clases
import cv2
from keras.models import model_from_json
import json
import keras 
from keras.initializers import glorot_uniform
from keras import initializers
from keras.models import load_model
import tensorflow as tf


CLASSES = ['cubism', 'expressionism', 'fauvism', 'pointillism','pop_art','ukiyo']
model = 'Modelo_entrenado/Checkpoint_12_0.87'

# esto no lo puedo hacer por los problemas con el bug de tensorflow con keras
def pred(path):
    im = cv2.imread(path)
    pred = model.predict(np.expand_dims(imres,axis=0))[0]
    for i, p in enumerate(pred):
        if p > 0.5:
            return {CLASSES[i]: str(p)}

pred('Modelo_entrenado/Checkpoint_257_0.87.h5')