import os
import pickle
import numpy as np
from utils import get_dir_list, save_pkl


# 获取文件中的landmarks值
def get_landmarks_data(file_name):
    data = []
    file = open(file_name, 'r')  # 打开文件
    file_data = file.readlines()  # 读取所有行
    for row in file_data:
        row = row.replace('\n', '')
        tmp_list = row.split(' ')  # 按‘ '切分每行的数据
        data.append([float(tmp_list[3]), float(tmp_list[6])])  # 将每行数据插入data中
    return data


# 获取文件中的label值
def get_label(file_name):
    f = open(file_name, 'r+')
    line = f.readline()  # only one row
    line_data = line.split(' ')
    label = float(line_data[3])
    f.close()
    # 1-7 的标签值转为 0-6
    return int(label) - 1


def main():
    pkl_dict = {"split": {}, "annotations": []}
    pkl_dict["split"] = {"train": [], "val": []}

    landmarks_root_dir = r'E:\PycharmProjects\dataset\CK+\Landmarks'
    label_root_dir = r'E:\PycharmProjects\dataset\CK+\Emotion'
    dir_list = get_dir_list(landmarks_root_dir)
    # label_dic = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Neutral": 4, "Sad": 5, "Surprise": 6}
    # classes = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
    all_dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
    fold_dic_10 = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for sub_dir in dir_list:
        landmarks_dir = landmarks_root_dir + "\\" + sub_dir
        label_dir = label_root_dir + "\\" + sub_dir
        landmarks_data = []
        label = None
        total_frame = 0
        flag = 0

        for _, _, files in os.walk(landmarks_dir):
            i = 1
            sub_dir_list = sub_dir.split("\\")
            data_file_tmp = sub_dir_list[0] + "_" + sub_dir_list[1] + "_" + "00000000" + "_landmarks.txt"
            while i <= len(files):
                replace_tmp = str(i) + "_landmarks.txt"
                replace_tmp_0 = "0" * len(str(i)) + "_landmarks.txt"
                data_file = data_file_tmp.replace(replace_tmp_0, replace_tmp)
                if data_file in files:
                    i += 1
                    total_frame += 1
                    landmarks_data.append(get_landmarks_data(landmarks_dir + '\\' + data_file))
                else:
                    i += 1
                    print("error" + data_file + " not in files  " + landmarks_dir)

        for _, _, files in os.walk(label_dir):
            if len(files) > 0:  # picture has label
                flag = 1
                label = get_label(label_dir + '\\' + files[0])
        if flag:
            all_dic[label].append(sub_dir)

            tmp_dict = {}
            tmp_dict["frame_dir"] = sub_dir
            tmp_dict["label"] = np.int64(label)
            tmp_dict["img_shape"] = (640, 490)
            tmp_dict["original_shape"] = (640, 490)
            tmp_dict["total_frames"] = total_frame
            tmp_dict["keypoint"] = np.array([landmarks_data])
            tmp_dict["keypoint_score"] = np.ones((1, total_frame, 68))

            pkl_dict["annotations"].append(tmp_dict)

    for i in range(10):
        for label_key in all_dic.keys():
            length = len(all_dic[label_key])
            fold_dic_10[i] = fold_dic_10[i] + all_dic[label_key][int((length/10)*i):int((length/10)*(i+1))]

    for i in range(10):
        print(len(fold_dic_10[i]))

    for i in range(10):
        pkl_dict["split"]["train"] = []
        pkl_dict["split"]["val"] = []
        key_list = list(fold_dic_10.keys())
        key_list.remove(i)
        for key in key_list:
            pkl_dict["split"]["train"] = pkl_dict["split"]["train"] + fold_dic_10[key]
        pkl_dict["split"]["val"] = fold_dic_10[i]
        save_pkl("..\\..\\data\\ck+_10_fold\\ck_landmarks_" + str(i) + ".pkl", pkl_dict)


if __name__=="__main__":
    main()