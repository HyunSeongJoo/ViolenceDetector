from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Conv2D, Reshape, Dropout, Add, MaxPooling3D, GlobalMaxPooling3D
from keras.layers import Dropout, Input, BatchNormalization, TimeDistributed,Activation
from sklearn.metrics import confusion_matrix, accuracy_score
# from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
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

def generate_data(directory, batch_size):
    i = 0

    file_list = os.listdir(directory)

    while True:
        X_batch = []
        y_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i =0
                random.shuffle(file_list)
            video_name = file_list[i]
            print('\r load video {} '.format( video_name),end='')
            input_seq = list()
            i += 1
            cap_train = cv2.VideoCapture(os.path.join(directory, video_name))

            # 답 저장
            if(video_name.split('_')[0] == 'NV'):
                y_batch.append(0)
            elif (video_name.split('_')[0] == 'V'):
                y_batch.append(1)

            # 정해진 개수 씩 읽어다가 특징 추출 할거다
            frame_count = shape[0]
            # 정해진 개수 씩 읽기위한 트리거F
            trigger_train = 0
            # 일정 간격으로 프레임 선정
            gap = int(cap_train.get(cv2.CAP_PROP_FRAME_COUNT) / shape[0])
            frame_num_list = [int(x * gap)+1 for x in range(shape[0])]
            while (cap_train.isOpened()):
                ret, frame = cap_train.read()
                if trigger_train in frame_num_list:
                    frame = cv2.resize(frame, (shape[1], shape[2]))
                    input_seq.append(frame)
                    if len(input_seq) == frame_count:
                        # 트리거 초기화
                        trigger_train = 0
                        # 프레임을 np.array로 변환 후 저장
                        X_batch.append(np.array(input_seq))
                        input_seq = []
                        break
                trigger_train += 1
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

            cap_train.release()
            cv2.destroyAllWindows()
            # if len(input_stack) == count_V or len(input_stack) == count_V + count_VN:
            #     # if len(input_stack)%800 == 0:
            #     break

        X_batch = np.array(X_batch)
        y_batch = to_categorical(np.array(y_batch), num_classes=2)
        # y_batch = np.array(y_batch).reshape(-1, 1)
        # enc = OneHotEncoder()
        # enc.fit(y_batch)
        # y_batch = enc.transform(y_batch).toarray()

        yield X_batch, y_batch

def conv2d_module(input_shape):

    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    vgg_model.trainable=False
    video = Input(shape=input_shape)

    cnn_out = vgg_model.output
    flatten_vec=Flatten()(cnn_out)
    dense_1=Dense(1024)(flatten_vec)
    cnn = Model(input=vgg_model.input, output=dense_1)

    encoded_frames = TimeDistributed(cnn)(video)
    print()
    print(encoded_frames)
    encoded_sequence = keras.layers.LSTM(256)(encoded_frames)
    hidden_layer = Dense(output_dim=1024, activation='relu')(encoded_sequence)
    outputs = Dense(output_dim=2, activation='softmax')(hidden_layer)
    custom_model = Model(input=video, output=outputs)


    custom_model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])
    # # x = keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
    return custom_model

def conv3d_LSTM_module(input_shape):#(10,3,224,224,3)
    def conv3d_module(input_layer,filters, kernel_size, activation,padding):
        conv_layer1 = Conv3D(filters=filters, kernel_size=kernel_size, activation=activation,padding=padding)(input_layer)
        conv_layer2 = Conv3D(filters=filters, kernel_size=kernel_size, activation=activation,padding=padding)(conv_layer1)

        pooling_output = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)
        return pooling_output
    input_layer = Input(shape=input_shape)

    pooling_layer1 = conv3d_module(input_layer, 8, (3,3,3), 'relu','same')
    print(pooling_layer1.shape)
    pooling_layer2 = conv3d_module(pooling_layer1, 16, (3,3,3), 'relu','same')
    print(pooling_layer2.shape)
    pooling_layer3 = conv3d_module(pooling_layer2, 32, (3,3,3), 'relu','same')
    #pooling_layer3 = BatchNormalization()(pooling_layer2)
    print(pooling_layer3.shape)
    flatten_vec=Reshape((pooling_layer3.shape[1],pooling_layer3.shape[2]*pooling_layer3.shape[3]*pooling_layer3.shape[4]))(pooling_layer3)

    print(flatten_vec.shape)
    # dense_1=Dense(1024)(flatten_vec)
    c3d = Model(input=input_layer, output=flatten_vec)

    encoded_frames = TimeDistributed(c3d)(input_layer)
    print(encoded_frames.shape)
    #print(encoded_frames)
    encoded_sequence = keras.layers.LSTM(256)(encoded_frames)
    hidden_layer = Dense(output_dim=1024, activation='relu')(encoded_sequence)
    outputs = Dense(output_dim=2, activation='softmax')(hidden_layer)
    custom_model = Model(input=input_layer, output=outputs)


    custom_model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])
    return custom_model

def model_build(input_shape):
    # 입력 받는 레이어에 입력 형태 설정
    def conv3d_module(input_layer,filters, kernel_size, activation):
        conv_layer1 = Conv3D(filters=filters, kernel_size=kernel_size, activation=activation)(input_layer)
        conv_layer2 = Conv3D(filters=filters, kernel_size=kernel_size, activation=activation)(conv_layer1)

        pooling_output = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)
        return pooling_output



    # def conv3d_RNN_module():

    input_layer = Input(shape=input_shape)

    pooling_layer1 = conv3d_module(input_layer, 8, (3,3,3), 'relu')

    pooling_layer2 = conv3d_module(pooling_layer1, 16, (3,3,3), 'relu')

    conv_layer5 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer2)
    conv_layer6 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(conv_layer5)

    pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer6)

    pooling_layer3 = BatchNormalization()(pooling_layer3)
    flatten_layer = Flatten()(pooling_layer3)

    # dense_layer1 = Dense(units=512, activation='relu')(flatten_layer)
    # dropout 이 좋은 건 알겠지만 0.4 는 그냥 따라 쓴것
    # dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    dense_layer3 = Dense(units=64, activation='relu')(dense_layer2)
    dense_layer3 = Dropout(0.4)(dense_layer3)
    output_layer = Dense(units=2, activation='softmax')(dense_layer3)

    model = Model(inputs=input_layer, outputs=output_layer)
    print('build done')
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.1), metrics=['acc'])

    return model
def C3D_N_Res_model_build(input_shape):
    # 입력 받는 레이어에 입력 형태 설정
    def conv3d_module(input_layer,filters, kernel_size, activation,padding):

        conv_layer1 = Conv3D(filters=filters, kernel_size=kernel_size,padding=padding)(input_layer)
        bn_layer1 = BatchNormalization()(conv_layer1)
        ac_layer1=Activation(activation)(bn_layer1)
        dr_layer1=Dropout(0.5)(ac_layer1)
        conv_layer2 = Conv3D(filters=filters, kernel_size=kernel_size,padding=padding)(dr_layer1)
        bn_layer2 = BatchNormalization()(conv_layer2)
        ac_layer2 = Activation(activation)(bn_layer2)

        res_out=Add()([input_layer, ac_layer2])

        pooling_output = MaxPool3D(pool_size=(2, 2, 2))(res_out)
        return pooling_output



    # def conv3d_RNN_module():

    input_layer = Input(shape=input_shape)

    pooling_layer1 = conv3d_module(input_layer, 64, (3,3,3), 'relu','same')

    pooling_layer2 = conv3d_module(pooling_layer1, 64, (3,3,3), 'relu','same')

    pooling_layer3 = conv3d_module(pooling_layer2, 64, (3, 3, 3), 'relu', 'same')

    # pooling_layer3 = BatchNormalization()(pooling_layer3)
    flatten_layer = Flatten()(pooling_layer3)

    # dense_layer1 = Dense(units=512, activation='relu')(flatten_layer)
    # dropout 이 좋은 건 알겠지만 0.4 는 그냥 따라 쓴것
    # dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    dense_layer3 = Dense(units=64, activation='relu')(dense_layer2)
    dense_layer3 = Dropout(0.4)(dense_layer3)
    output_layer = Dense(units=2, activation='softmax')(dense_layer3)

    model = Model(inputs=input_layer, outputs=output_layer)
    print('build done')
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.1), metrics=['acc'])

    return model
def sungmin_model(input_shape):
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
    added = GlobalMaxPooling3D(pool_size=(2, 2, 2), padding='same')(added)
    added = Flatten()(added)

    output_layer = Dropout(0.4)(added)

    model = Model(input_layer, output_layer)

    model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['accuracy'])
    print('build done')
    return model

if __name__ == "__main__":
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
    model = C3D_N_Res_model_build(shape)
    model.summary()
    if phase == 'Train':
        y_train = list()
        # 30frame 이상쓰면 리소스 부족..
        # model = model_build(shape)
        # model = conv2d_module(shape)
        # model = conv3d_LSTM_module(shape2)
        model = C3D_N_Res_model_build(shape)
        # model = sungmin_model(shape)
        c3d_LSTM = False

        model.summary()
        # 체크포인트 저장 경로
        checkpoint_path = "training_1/200610_3_fit.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # 체크포인트 콜백
        cp_callback = [
            keras.callbacks.EarlyStopping(patience=3),
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        ]

        batch_size = 2

        if c3d_LSTM is False:

            # 데이터 불러오기
            for folder_name in folder_list_train:
                for video_name in os.listdir(os.path.join('.',folder_name)):
                    print('load video ', video_name)
                    cap = cv2.VideoCapture(os.path.join('.', folder_name, video_name))
                    # 답 저장
                    y_train.append(video_name.split('_')[0])
                    # 정해진 개수 씩 읽어다가 특징 추출 할거다
                    input_sequence_length = shape[0]
                    # 정해진 개수 씩 읽기위한 트리거
                    trigger = 0
                    # 일정 간격으로 프레임 선정
                    gap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / shape[0])
                    frame_num_list = [int(x * gap) for x in range(shape[0])]
                    while(cap.isOpened()):
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
                    if len(input_stack) == count_VN+count_V:
                        break


            input_stack = np.array(input_stack)
            y_train = np.array(y_train).reshape(-1,1)
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
                            if three %3 == 0:
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

        # model.fit(
        #     x=input_stack,
        #     y=y_train,
        #     batch_size=1,
        #     epochs=50,
        #     validation_split=0.2,
        #     callbacks=[cp_callback])
        # X_train, X_test, Y_train, Y_test = train_test_split(input_stack, y_train, test_size=0.25, random_state=321)

        model.fit(x=input_stack, y=y_train, batch_size=1, epochs=100, validation_split=0.2, callbacks=cp_callback, verbose=2)
    else:
        folder_list_test = ['splited_data/test/V', 'splited_data/test/NV']
        count_V_test = 249
        count_VN_test = 232
        y_test=list()

        model = model_build(shape)
        # model = conv2d_module(shape)
        model.load_weights('training_1/200608_1_fit.ckpt')

        for folder_name in folder_list_test:
            for video_name in os.listdir(os.path.join('.',folder_name)):
                print('load video ', video_name)
                cap = cv2.VideoCapture(os.path.join('.', folder_name, video_name))
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
                while(cap.isOpened()):
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
                if len(input_stack) == count_V_test or len(input_stack) == count_V_test+count_VN_test:
                # if len(input_stack)%800 == 0:
                    break

        input_stack = np.array(input_stack)
        y_test = np.array(y_test).reshape(-1,1)
        enc = OneHotEncoder()
        enc.fit(y_test)
        y_test = enc.transform(y_test).toarray()

        #pred_y = model.predict(input_stack)

        # acc = accuracy(y_test,pred_y)
        # print(acc)
        results=model.evaluate(input_stack, y_test, batch_size=1)
        print('test loss, test acc:', results)

# results = model.evaluate(X_test, Y_test)
# print(results)

# test loss, test acc: [110.90682757059629, 0.42203742265701294]   20.06.03