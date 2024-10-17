import os
import pickle
import numpy as np
import cv2


def get_dir_list(root_dir):
    # 获取子目录列表, like ['S005\001', 'S010\001', 'S010\002', ...]
    dir_list = []
    for root, dirs, _ in os.walk(root_dir):
        for rdir in dirs:
            for _, sub_dirs, _ in os.walk(root + '\\' + rdir):
                for sub_dir in sub_dirs:
                    dir_list.append(rdir + '\\' + sub_dir)
    return dir_list


def save_pkl(file_name, data):
    f = open(file_name, 'wb')
    pickle.dump(data, f)
    f.close()


def read_pickle(work_path):
    with open(work_path, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
            except EOFError:
                break
    return data


def claheColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def LinearEqual(image):
    lut = np.zeros(256, dtype=image.dtype)
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minBinNo, maxBinNo = 0, 255

    for binNo, binValue in enumerate(hist):
        if binValue != 0:
            minBinNo = binNo
            break
    for binNo, binValue in enumerate(reversed(hist)):
        if binValue != 0:
            maxBinNo = 255 - binNo
            break
    for i, v in enumerate(lut):
        # print(i)
        if i < minBinNo:
            lut[i] = 0
        elif i > maxBinNo:
            lut[i] = 255
        else:
            lut[i] = int(255.0 * (i - minBinNo) / (maxBinNo - minBinNo) + 0.5)  # why plus 0.5
    return cv2.LUT(image, lut)


def write_txt_a(file_name, content):
    with open(file_name, "a") as file:
        file.write(content + "\n")
