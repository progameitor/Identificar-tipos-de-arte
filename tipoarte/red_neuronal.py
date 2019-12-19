# preparo el modelo para entrenarlo
X_train, X_test, y_train, y_test = train_test_split(X, array_y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# shape and number of classes
num_classes = 6
input_shape = (128, 128, 3)

#NN topology( la pongo aqui para tenerla localizada)

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
model.compile(loss=tf.losses.categorical_crossentropy,
              optimizer=tf.optimizers.Adadelta(),
              metrics=['accuracy'])

# para guardar los checkpoitns cuando el accuracy de mi modelo sube
filepath='Checkpoint_{epoch:02d}_{val_acc:.2f}'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# compruebo la forma para ver que es correcta al meterlo en la red neuronal
print(X_train.shape, y_train.shape)

# parámetros de la red neuronal
batch_size = 125
epochs = 800

history =model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=callbacks_list)

# Evaluate the model with test data (me evalua mi prediccion)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])