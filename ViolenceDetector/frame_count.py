import cv2
import os


folder_list = ['splited_data/train', 'splited_data/test']
# for i in os.listdir('.'):
#     if os.path.isdir(i):
#         folder_list.append(i)




NV_train_list = list()
NV_test_list = list()
V_train_list = list()
V_test_list = list()
frames = list()
videos = [[] for _ in range(20000)]
for folder_name in folder_list:
    for video_name in os.listdir(os.path.join('.',folder_name)):
        if 'avi' in video_name:

            cap = cv2.VideoCapture(os.path.join('.',folder_name,video_name))

            frames.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            videos[int(cap.get(cv2.CAP_PROP_FRAME_COUNT))].append(video_name)
            if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 121:
                if video_name.split('_')[0] == 'NV':
                    if 'train' in folder_name:
                        NV_train_list.append(video_name)
                    else:
                        NV_test_list.append(video_name)
                elif video_name.split('_')[0] == 'V':
                    if 'train' in folder_name:
                        V_train_list.append(video_name)
                    else:
                        V_test_list.append(video_name)
            # print(video_name, ' done.')


# print('NV_train')
# for i in NV_train_list:
#     print(i)

# print('NV_test')
# for i in NV_test_list:
#     print(i)

# print('V_train')
# for i in V_train_list:
#     print(i)

print('V_test')
for i in V_test_list:
    print(i)

print('NV_train ', len(NV_train_list))
print('NV_test ', len(NV_test_list))
print('V_train ', len(V_train_list))
print('V_test ', len(V_test_list))

dst_list = ['data_121/train', 'data_121/test']


# print('NV_train ', len(NV_train_list), NV_train_list)
# print('NV_test ', len(NV_test_list), NV_test_list)
# print('V_train ', len(V_train_list), V_train_list)
# print('V_test ', len(V_test_list), V_test_list)

# count_nv = 0
# count_v = 0
# NV_list = list()
# V_list = list()
# # for frame, i in enumerate(videos):
# #     if len(i) != 0:
# #         print(frame, ' ', len(i), ' ',  i)
#
# for video in videos[121]:
#     if video.split('_')[0] == 'V':
#         count_v += 1
#         V_list.append(video)
#     else:
#         count_nv += 1
#         NV_list.append(video)
#         # print(video)
#     # print(video)
#
# print(V_list)
# print(NV_list)
# print(count_v)
# print(count_nv)
# print('max ', max(frames))
# print('min ', min(frames))
# print('average ', sum(frames)/len(frames))



# 비디오 불러옴

# cap = cv2.VideoCapture('V_17.mp4')

# root_path = 'C:\\Users\\HyunSeong\\Desktop\\Real Life Violence Dataset'
# input_folder = 'NV_avi_rename'
# train_folder = 'splited_data\\train\\NV'
# test_folder =  'splited_data\\test\\NV'
#
# number = 1
# for video_name in os.listdir(os.path.join(root_path, input_folder)):
#     # new_name = '{}_{}.avi'.format(video_name.split('_')[0],number)
#     if number % 4 ==0 :
#         os.rename(os.path.join(root_path, input_folder, video_name), os.path.join(root_path, test_folder, video_name))
#     else:
#         os.rename(os.path.join(root_path,input_folder,video_name), os.path.join(root_path,train_folder,video_name))
#     number+=1
#     # print(video_name , " => ", new_name)