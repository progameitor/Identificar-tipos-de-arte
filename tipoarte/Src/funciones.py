import glob
import json
from keras.models import model_from_json
import cv2
import numpy as np
from keras.models import load_model


# saco las rutas de mi carpeta
def obtener_Ruta(carpeta):
    first = 'Inputs/imagenes_extraidas/128x128/*'
    last = '*.jpg'
    result = first + carpeta + last
    return result

diccionario = {'cubism_128/': 1,
               'expressionism_128/':2,
               "fauvism_128_espejo/":3,
               "pointillism_126_espejo/":4,
               "pop_art_128_espejo/":5,
               'ukiyo_128/':6            
               }
# obtengot todas las imagenes de mis carpetas y las appendo un valor
def sacar_Imagenes(carpeta):
    lista = []
    for x in glob.glob(obtener_Ruta(carpeta)):
        for k,v in diccionario.items():
            if k in x:
                lista.append(v)
    return lista


# aqui me sirve para plotear como iba el avance de mi red neuronal
def plotting_train(history):
    ''' Plots showing losses and accuracy of the model'''
    #Plotting params
    acc = history.history['accuracy']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()
    plt.figure()
    # Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()

# parametros necesarios para la funcion de predict
img_route = "/content/drive/My Drive/Test-photos/test4.jpg"
model_json = '/content/model_071_01.json'
weights = '/content/Checkpoint_257_0.73'
CLASSES = ['cubism', 'expressionism', 'fauvism', 'pointillism','pop_art','ukiyo']

# aqui sacamos las precidiones del modelo    
def predict(image_dir,model_dir,weight_dir):
    '''Predict the disease of the given photo'''
    # load json and create model
    with open(model_dir,'r') as f:
        model_json = json.load(f)
    loaded_model = model_from_json(model_json)    
    print("Loaded model from disk")
    loaded_model.load_weights(weights)
    #image processing and recognition
    default_image_size = tuple((128, 128))
    img = cv2.imread(image_dir) / 255
    img = cv2.resize(img, default_image_size)
    np_image_li = np.asarray(img)
    npp_image = np.expand_dims(np_image_li, axis=0)
    result=loaded_model.predict(npp_image)
    itemindex = np.where(result==np.max(result))
    print('\n' + "Probability: "+str(np.max(result)) + ' ====> ' + CLASSES[itemindex[1][0]] + '\n')
    return result     