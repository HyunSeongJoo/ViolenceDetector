from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
# from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adadelta
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


def model_build(input_shape):
    # 입력 받는 레이어에 입력 형태 설정
    input_layer = Input(shape=input_shape)

    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

    conv_layer3 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)

    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

    # conv_layer5 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer2)
    # conv_layer6 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu')(conv_layer5)

    # pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer5)

    pooling_layer3 = BatchNormalization()(pooling_layer2)
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
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])

    return model

if __name__ == "__main__":
    video_total_dir = 'C:/Users/HyunSeong/Desktop/Real Life Violence Dataset'
    video_V_folder = '/splited_data/train/V'
    video_NV_folder = '/splited_data/train/NV'

    folder_list_train = ['splited_data/train/V', 'splited_data/train/NV']
    count_V = 748
    count_VN = 698
    phase = 'Train'

    # for i in os.listdir('.'):
    #     if os.path.isdir(i):
    #         folder_list.append(i)
    shape = (30, 224, 224, 3)

    # input_sequence = list()
    input_stack = list()

    if phase == 'Train':
        y_train = list()
        # 30frame 이상쓰면 리소스 부족..
        model = model_build(shape)
        model.summary()
        # 체크포인트 저장 경로
        checkpoint_path = "training_1/200604_3_split_data.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # 체크포인트 콜백
        cp_callback = [
            keras.callbacks.EarlyStopping(patience=5),
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        ]

        batch_size = 2

        model.fit_generator(
            generate_data('./splited_data/train', batch_size),
            epochs=8,
            steps_per_epoch=(count_V+count_VN)/batch_size, verbose=2,
            callbacks=cp_callback,
            validation_data=(generate_data('./splited_data/train', batch_size)),
            validation_steps=((count_V+count_VN)*0.25),
            workers=6,
            use_multiprocessing=True
        )
        #
        #         # model.fit(x=input_stack, y=y_train, batch_size=1, epochs=100, validation_split=0.2, callbacks=cp_callback)

        # # 데이터 불러오기
        # for folder_name in folder_list_train:
        #     for video_name in os.listdir(os.path.join('.',folder_name)):
        #         print('load video ', video_name)
        #         cap = cv2.VideoCapture(os.path.join('.', folder_name, video_name))
        #         # 답 저장
        #         y_train.append(video_name.split('_')[0])
        #         # 정해진 개수 씩 읽어다가 특징 추출 할거다
        #         input_sequence_length = shape[0]
        #         # 정해진 개수 씩 읽기위한 트리거
        #         trigger = 0
        #         # 일정 간격으로 프레임 선정
        #         gap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / shape[0])
        #         frame_num_list = [int(x * gap) for x in range(shape[0])]
        #         while(cap.isOpened()):
        #             ret, frame = cap.read()
        #             if trigger in frame_num_list:
        #                 frame = cv2.resize(frame, (shape[1], shape[2]))
        #                 input_sequence.append(frame)
        #                 if len(input_sequence) == input_sequence_length:
        #                     # 트리거 초기화
        #                     trigger = 0
        #                     # 프레임을 np.array로 변환 후 저장
        #                     input_stack.append(np.array(input_sequence))
        #                     input_sequence = list()
        #                     break
        #             trigger += 1
        #             if cv2.waitKey(1) & 0xFF == ord('q'):
        #                 break
        #
        #         cap.release()
        #         cv2.destroyAllWindows()
        #         if len(input_stack) == count_V or len(input_stack) == count_V+count_VN:
        #         # if len(input_stack)%800 == 0:
        #             break
        #
        #
        # input_stack = np.array(input_stack)
        # y_train = np.array(y_train).reshape(-1,1)
        # enc = OneHotEncoder()
        # enc.fit(y_train)
        # y_train = enc.transform(y_train).toarray()


        # model.fit(
        #     x=input_stack,
        #     y=y_train,
        #     batch_size=1,
        #     epochs=50,
        #     validation_split=0.2,
        #     callbacks=[cp_callback])
        # X_train, X_test, Y_train, Y_test = train_test_split(input_stack, y_train, test_size=0.25, random_state=321)

        # model.fit(x=input_stack, y=y_train, batch_size=1, epochs=100, validation_split=0.2, callbacks=cp_callback)
    else:
        folder_list_test = ['splited_data/test/V', 'splited_data/test/NV']
        count_V_test = 249
        count_VN_test = 232
        y_test=list()

        model = model_build(shape)
        model.load_weights('training_1/200601_1_split_data.ckpt')

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