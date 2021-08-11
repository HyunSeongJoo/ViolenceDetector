# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:49:18 2020

@author: admin
"""

from keras.layers import Conv1D, Conv3D, MaxPooling3D, BatchNormalization, Dense, Dropout, Activation, Input, Add, \
    GlobalAveragePooling3D
from keras.models import Model
import os
import keras
import cv2
# from plotly.offline import iplot, init_notebook_mode
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
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
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16


def model_build(input_shape):
    def split_3D_module(_input, filters, kernel_size, padding, activation):
        conv1 = Conv3D(filters=filters, kernel_size=(1, 3, 3), padding=padding, activation=activation)(_input)
        conv2_1 = Conv3D(filters=filters, kernel_size=(3, 3, 1), padding=padding, activation=activation)(conv1)
        conv2_2 = Conv3D(filters=filters, kernel_size=(3, 1, 3), padding=padding, activation=activation)(conv1)
        conv3 = Add()([conv2_1, conv2_2])
        return conv3

    def conv3D_module(_input, filters, kernel_size, padding, activation):
        conv_1 = split_3D_module(_input, filters, kernel_size, padding, activation)
        Bn_1 = BatchNormalization()(conv_1)
        Ac_1 = Activation(activation)(Bn_1)
        out_1 = Dropout(0.5)(Ac_1)
        conv_2 = split_3D_module(out_1, filters, kernel_size, padding, activation)
        Bn_2 = BatchNormalization()(conv_2)
        Ac_2 = Activation(activation)(Bn_2)
        if _input.shape[-1] == filters:
            res_out = Add()([_input, Ac_2])
        else:
            res_out = Ac_2
        return res_out

    input_layer = Input(shape=input_shape, name="main_input")
    Bn = BatchNormalization()(input_layer)
    Bn = split_3D_module(Bn, 4, (3, 3, 3), 'same', 'relu')
    # 8 16 32 가능
    block_1 = conv3D_module(Bn, 4, (3, 3, 3), 'same', 'relu')

    block_2 = conv3D_module(block_1, 8, (3, 3, 3), 'same', 'relu')

    block_3 = conv3D_module(block_2, 16, (3, 3, 3), 'same', 'relu')

    block_4 = conv3D_module(block_3, 32, (3, 3, 3), 'same', 'relu')

    # block_5 = conv3D_module(block_4, 128, (3, 3, 3), 'same', 'relu')

    # block_6 = conv3D_module(block_5, 16, (3, 3, 3), 'same', 'relu')

    # block_7 = conv3D_module(block_6, 32, (3, 3, 3), 'same', 'relu')
    #
    # block_8 = conv3D_module(block_7, 32, (3, 3, 3), 'same', 'relu')
    #
    # block_9 = conv3D_module(block_8, 32, (3, 3, 3), 'same', 'relu')

    GAP = GlobalAveragePooling3D()(block_4)

    output = Dense(units=2, activation='softmax')(GAP)


    return Model(input_layer, output)


if __name__ == "__main__":
    input_shape = (30, 224, 224, 3)
    model = model_build(input_shape)
    model.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])

    model.summary()

    video_total_dir = 'C:/Users/HyunSeong/Desktop/Real Life Violence Dataset'
    video_V_folder = '/splited_data/train/V'
    video_NV_folder = '/splited_data/train/NV'

    folder_list_train = ['splited_data/train']
    count_V = 748
    count_VN = 698
    phase = 'Train'

    # for i in os.listdir('.'):
    #     if os.path.isdir(i):
    #         folder_list.append(i)
    shape = (30, 224, 224, 3)
    shape2 = (10, 3, 224, 224, 3)
    input_stack = list()
    input_stack2 = list()
    input_sequence = list()

    if phase == 'Train':
        y_train = list()
        # 30frame 이상쓰면 리소스 부족..
        # model = model_build(shape)
        # model = conv2d_module(shape)
        # model = conv3d_LSTM_module(shape2)
        # model = C3D_N_Res_model_build(shape)
        # model = sungmin_model(shape)
        c3d_LSTM = False

        # model.summary()
        # 체크포인트 저장 경로
        checkpoint_path = "training_1/200616_4_deep.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # 체크포인트 콜백
        cp_callback = [
            keras.callbacks.EarlyStopping(patience=3),
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=3, min_lr=0.001)

        ]

        batch_size = 2

        if c3d_LSTM is False:

            # 데이터 불러오기
            for folder_name in folder_list_train:
                for i,video_name in enumerate(os.listdir(os.path.join('.', folder_name))):
                    print('{}th load video : {}'.format(i, video_name),end=' ')
                    cap = cv2.VideoCapture(os.path.join('.', folder_name, video_name))
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if length!=121:
                        print('not 121 => {}'.format(length),end='\n')
                        cap.release()
                        cv2.destroyAllWindows()
                        continue
                    else:
                        print('121 frame',end='\n')
                        pass
                    # 답 저장
                    y_train.append(video_name.split('_')[0])
                    # 정해진 개수 씩 읽어다가 특징 추출 할거다
                    input_sequence_length = shape[0]
                    # 정해진 개수 씩 읽기위한 트리거
                    trigger = 0
                    # 일정 간격으로 프레임 선정
                    gap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / shape[0])
                    frame_num_list = [int(x * gap) for x in range(shape[0])]
                    while (cap.isOpened()):
                        ret, frame = cap.read()
                        if trigger in frame_num_list:
                            frame = cv2.resize(frame, (shape[1], shape[2]))
                            input_sequence.append(frame)
                            if len(input_sequence) == input_sequence_length:
                                # 트리거 초기화
                                trigger = 0
                                # 프레임을 np.array로 변환 후 저장
                                input_stack.append(np.array(input_sequence))
                                input_sequence = list()
                                break
                        trigger += 1
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cap.release()
                    cv2.destroyAllWindows()
                    # if len(input_stack) == count_V or len(input_stack) == count_V+count_VN:
                    if len(input_stack) == count_VN + count_V:
                        break

            input_stack = np.array(input_stack)
            y_train = np.array(y_train).reshape(-1, 1)
            enc = OneHotEncoder()
            enc.fit(y_train)
            y_train = enc.transform(y_train).toarray()
        else:
            # 데이터 불러오기
            for folder_name in folder_list_train:
                for video_name in os.listdir(os.path.join('.', folder_name)):
                    print('load video ', video_name)
                    cap = cv2.VideoCapture(os.path.join('.', folder_name, video_name))
                    # 답 저장
                    y_train.append(video_name.split('_')[0])
                    # 정해진 개수 씩 읽어다가 특징 추출 할거다
                    input_sequence_length = shape2[0]
                    # 정해진 개수 씩 읽기위한 트리거
                    trigger = 0
                    three = 0
                    # 일정 간격으로 프레임 선정
                    gap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / shape[0])
                    frame_num_list = [int(x * gap) for x in range(shape[0])]
                    while (cap.isOpened()):
                        ret, frame = cap.read()
                        if trigger in frame_num_list:
                            frame = cv2.resize(frame, (shape[1], shape[2]))
                            input_sequence.append(frame)
                            three += 1
                            if three % 3 == 0:
                                three = 0
                                input_stack2.append(np.array(input_sequence))
                                input_sequence = list()

                            if len(input_stack2) == input_sequence_length:
                                # 트리거 초기화
                                trigger = 0
                                # 프레임을 np.array로 변환 후 저장
                                input_stack.append(np.array(input_stack2))
                                input_stack2 = list()
                                break
                        trigger += 1

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cap.release()
                    cv2.destroyAllWindows()
                    # if len(input_stack) == count_V or len(input_stack) == count_V+count_VN:
                    if len(input_stack) == count_VN + count_V:
                        break

            input_stack = np.array(input_stack)
            y_train = np.array(y_train).reshape(-1, 1)
            enc = OneHotEncoder()
            enc.fit(y_train)
            y_train = enc.transform(y_train).toarray()

        X_train, X_val, Y_train, Y_val = train_test_split(input_stack, y_train, test_size=0.2, random_state=321)

        # model.fit(
        #     x=X_train,
        #     y=Y_train,
        #     batch_size=1,
        #     epochs=50,
        #     validation_data=(X_val,Y_val),
        #     callbacks=[cp_callback])

        # model.fit(x=input_stack, y=y_train, batch_size=1, epochs=100, validation_split=0.2, callbacks=cp_callback,
        #           verbose=1)
        model.fit(x=X_train, y=Y_train, batch_size=1, epochs=100, validation_data=(X_val, Y_val), callbacks=cp_callback,
                  verbose=1)

    else:
        folder_list_test = ['splited_data/test']
        count_V_test = 249
        count_VN_test = 232
        y_test = list()

        # model = model_build(shape)
        # model = conv2d_module(shape)
        # 체크포인트 불러오기
        model.load_weights('training_1/200612_3.ckpt')

        for folder_name in folder_list_test:
            for i, video_name in enumerate(os.listdir(os.path.join('.', folder_name))):
                print('{}th load video : {}'.format(i, video_name), end=' ')
                cap = cv2.VideoCapture(os.path.join('.', folder_name, video_name))
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if length != 121:
                    print('not 121 => {}'.format(length), end='\n')
                    cap.release()
                    cv2.destroyAllWindows()
                    continue
                else:
                    print('121 frame', end='\n')
                    pass

                # 답 저장
                y_test.append(video_name.split('_')[0])

                # 정해진 개수 씩 읽어다가 특징 추출 할거다
                input_sequence_length = shape[0]

                # 정해진 개수 씩 읽기위한 트리거
                trigger = 0

                # 일정 간격으로 프레임 선정
                gap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / shape[0])
                frame_num_list = [int(x * gap) for x in range(shape[0])]

                # print(frame_num_list)
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if trigger in frame_num_list:
                        frame = cv2.resize(frame, (shape[1], shape[2]))

                        input_sequence.append(frame)
                        if len(input_sequence) == input_sequence_length:
                            # 트리거 초기화
                            trigger = 0
                            # 프레임을 np.array로 변환 후 저장
                            input_stack.append(np.array(input_sequence))
                            input_sequence = list()
                            break
                    trigger += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
                if len(input_stack) == count_V_test or len(input_stack) == count_V_test + count_VN_test:
                    # if len(input_stack)%800 == 0:
                    break

        input_stack = np.array(input_stack)
        y_test = np.array(y_test).reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(y_test)
        y_test = enc.transform(y_test).toarray()

        # pred_y = model.predict(input_stack)

        # acc = accuracy(y_test,pred_y)
        # print(acc)
        results = model.evaluate(input_stack, y_test, batch_size=1)
        print('test loss, test acc:', results)
