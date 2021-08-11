from keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
# from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adadelta, adam
import plotly.graph_objs as go
from matplotlib.pyplot import cm
from keras.models import Model
import keras
import cv2
import numpy as np
import random
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as Keras_GPU
from sklearn.model_selection import train_test_split

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print('start!')
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)

'''




def model_build(input_shape):
    input_layer = Input(shape=input_shape)


    initial_conv = Conv3D(8, kernel_size=(3, 3, 3), padding='same', activation='relu')(input_layer)
    initial_conv = Conv3D(16, kernel_size=(3, 3, 3), padding='same', activation='relu')(initial_conv)

    # 2번째 conv layer

    conv2 = Conv3D(8, kernel_size=(1, 1, 1), padding='same', activation='relu')(initial_conv)
    conv2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    conv2 = Conv3D(8, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv2)
    conv2 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv2)
    conv2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', activation='relu')(conv2)
    ##############################

    # 3번째 conv layer

    conv3 = Conv3D(4, kernel_size=(1, 1, 1), padding='same', activation='relu')(initial_conv)
    conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
    conv3 = Conv3D(4, kernel_size=(3, 3, 3), padding='same', activation='relu')(conv3)
    conv3 = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conv3)
    conv3 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', activation='relu')(conv3)

    added = keras.layers.Add()([conv2, conv3])
    added = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(added)
    added = Flatten()(added)

    dense_1 = Dense(784)(added)
    dense_2 = Dense(196)(dense_1)

    output_layer = Dropout(0.4)(dense_2)

    model = Model(input_layer, output_layer)

    #model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['acc'])
    print('build done')
    return model

# 비디오의 형태 지정.
# 프레임 수
# max  11272.0
# min  49.0
# average  143.6845
# 30frame 이상쓰면 리소스 부족..

shape = (30, 224, 224, 3)

model = model_build(shape)
model.summary()
