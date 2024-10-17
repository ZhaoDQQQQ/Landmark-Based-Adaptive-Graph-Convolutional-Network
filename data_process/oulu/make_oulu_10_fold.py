import os
import pickle
import numpy as np
from utils import save_pkl


def get_dir_list(root_dir):
    # 获取子目录列表, like ['S005\001', 'S010\001', 'S010\002', ...]
    dir_list = []
    for root, dirs, _ in os.walk(root_dir):
        for rdir in dirs:
            for _, sub_dirs, _ in os.walk(root + '\\' + rdir):
                for sub_dir in sub_dirs:
                    dir_list.append(rdir + '\\' + sub_dir)
    return dir_list


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
def get_label(label_dic, file_name):
    emotion = file_name.split("\\")[-1]
    label = label_dic[emotion]
    return int(label)


def main():
    pkl_dict = {"split": {}, "annotations": []}
    pkl_dict["split"] = {"train": [], "val": []}

    landmarks_root_dir = r'E:\PycharmProjects\dataset\Oulu\dlib_landmarks'
    dir_list = get_dir_list(landmarks_root_dir)
    label_dic = {"Anger": 0, "Disgust": 1, "Fear": 2, "Happiness": 3, "Sadness": 4, "Surprise": 5}
    all_dic = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    fold_dic_10 = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for sub_dir in dir_list:
        landmarks_dir = landmarks_root_dir + "\\" + sub_dir
        landmarks_data = []
        label = None
        total_frame = 0

        for _, _, files in os.walk(landmarks_dir):
            i = 0
            while i < len(files):
                data_file = str(i) + "_my_landmarks.txt"
                if data_file in files:
                    i += 1
                    total_frame += 1
                    landmarks_data.append(get_landmarks_data(landmarks_dir + '\\' + data_file))
                else:
                    print("error:" + data_file + " not in files  " + landmarks_dir)

        label = get_label(label_dic, sub_dir)
        all_dic[label].append(sub_dir)

        tmp_dict = {}
        tmp_dict["frame_dir"] = sub_dir
        tmp_dict["label"] = np.int64(label)
        tmp_dict["img_shape"] = (320, 240)
        tmp_dict["original_shape"] = (320, 240)
        tmp_dict["total_frames"] = total_frame
        tmp_dict["keypoint"] = np.array([landmarks_data])
        tmp_dict["keypoint_score"] = np.ones((1, total_frame, 68))

        pkl_dict["annotations"].append(tmp_dict)

    # 因为oulu数据集的格式问题，会生成跨人的训练集、验证集    可以试试random
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
        save_pkl("..\\..\\data\\oulu_10_fold\\oulu_" + str(i) + ".pkl", pkl_dict)


if __name__=="__main__":
    main()