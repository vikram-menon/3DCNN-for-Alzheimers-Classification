import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

xtrain = np.load('xtrain.npy')
ytrain = np.load('ytrain.npy')

xtrain = xtrain/255.0

model = Sequential()
#first conv layer
model.add(Conv3D(8, kernel_size=(3, 3, 3), input_shape=(26, 48, 48, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.25))
#second conv layer
model.add(Conv3D(16, kernel_size=(3, 3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.25))
#third conv layer
model.add(Conv3D(16, kernel_size=(3, 3, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.25))

model.add(Flatten())

#first dense layer
model.add(Dense(16))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.25))
#second dense layer
model.add(Dense(16))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.25))
#third dense layer
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.25))
#fourth dense layer
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.summary()
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(xtrain, ytrain,
            batch_size=10,
            epochs=250,
            verbose=1,
            validation_split=0.3, 
            callbacks = [es_callback])
