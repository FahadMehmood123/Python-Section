from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2

########pip install keras==2.2.4
from keras.layers import Activation,  BatchNormalization , Conv2D, Input, MaxPooling2D, SeparableConv2D, GlobalAveragePooling2D
from keras.models import Model
from keras import layers
from keras.regularizers import l2


batch_size = 32
num_epochs = 50
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 3
base_path = './checkpoints/'

def modeldetection(inp_shape, target_classes, lr=0.01):
    rg = l2(lr)

    
    img_input = Input(inp_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=rg,
                                            use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=rg,
                                            use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # net 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=rg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=rg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # net 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=rg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=rg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # net 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=rg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=rg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # net 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=rg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=rg,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(target_classes, (3, 3),
            padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax',name='predictions')(x)

    model = Model(img_input, output)
    return model



# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)
						

# model parameters					
model = modeldetection(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


#trained model saving path and checkpoint
trained_models_path = base_path 
model_names = trained_models_path + '{epoch:02d}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                    save_best_only=True)

image_size=input_shape[:2]
path = './data.csv'
data = pd.read_csv(path)
pixels = data['pixels'].tolist()
width, height = 48, 48
emo_pics = []
for ps in pixels:
        emo_pic = [int(pixel) for pixel in ps.split(' ')]
        emo_pic = np.asarray(emo_pic).reshape(width, height)
        emo_pic = cv2.resize(emo_pic.astype('uint8'), image_size)
        emo_pics.append(emo_pic.astype('float32'))
emo_pics = np.asarray(emo_pics)
emo_pics = np.expand_dims(emo_pics, -1)
emotions = pd.get_dummies(data['emotion']).values

 			
emo_pics = emo_pics.astype('float32')
emo_pics = emo_pics / 255.0
emo_pics = emo_pics - 0.5
emo_pics = emo_pics * 2.0

num_samples, num_classes = emotions.shape
	
num_samples = len(emo_pics)
num_train_samples = int((1 - validation_split)*num_samples)
train_x = emo_pics[:num_train_samples]
train_y = emotions[:num_train_samples]
val_x = emo_pics[num_train_samples:]
val_y = emotions[num_train_samples:]
train_data = (train_x, train_y)
val_data = (val_x, val_y)
#training the model
train_emo_pics, train_emotions = train_data
m=model.fit_generator(data_generator.flow(train_emo_pics, train_emotions,
                                            batch_size),
                        steps_per_epoch=len(train_emo_pics) / batch_size,
                        epochs=num_epochs, verbose=1,callbacks=[model_checkpoint],
                        validation_data=val_data)
model.save('./model.h5')
import matplotlib.pyplot as plt
plt.plot(m.history['accuracy'])
plt.plot(m.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



