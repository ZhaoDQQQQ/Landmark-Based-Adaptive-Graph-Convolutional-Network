import cv2
import dlib
import time
from utils import claheColor, hisEqulColor, LinearEqual
import os
import sys


def get_dir_list(root_dir):
    # 获取子目录列表, like ['S005\001', 'S010\001', 'S010\002', ...]
    dir_list = []
    for root, dirs, _ in os.walk(root_dir):
        for rdir in dirs:
            for _, sub_dirs, _ in os.walk(root + '\\' + rdir):
                for sub_dir in sub_dirs:
                    dir_list.append(rdir + '\\' + sub_dir)
    return dir_list


def face_detector(bgr_img, detector, cnn_face_detector):
    # 转换成RGB图像
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    if len(dets) == 0:
        # CNN 人脸检测器
        dets = apply_cnn_detection(img, cnn_face_detector)
        if len(dets) == 0:
            img = LinearEqual(img)
            dets = apply_cnn_detection(img, cnn_face_detector)
            if len(dets) == 0:
                img = claheColor(img)
                dets = apply_cnn_detection(img, cnn_face_detector)
                if len(dets) == 0:
                    img = hisEqulColor(img)    # 直方图均衡化
                    dets = apply_cnn_detection(img, cnn_face_detector)
                    if len(dets) == 0:
                        return None
    return dets[0]


def apply_cnn_detection(img, cnn_face_detector):
    cnn_dets = cnn_face_detector(img, 1)
    dets = dlib.rectangles()
    dets.extend([d.rect for d in cnn_dets])
    return dets


def get_landmarks(img, face_rec, predictor_68):
    pt_position = []

    shape = predictor_68(img, face_rec)
    if len(shape.parts()) == 0:
        return None
    for pt in shape.parts():
        pt_position.append([pt.x, pt.y])
    return pt_position


def write_landmarks_txt(txt_name, landmarks_data):
    f = open(txt_name, 'w+')
    for pt in landmarks_data:
        f.write("   " + str(pt[0]) + "   " + str(pt[1]) + "\n")
    f.close()


def mkdir_or_exist(dir_name):
    if os.path.exists(dir_name):
        pass
    else:
        os.makedirs(dir_name)


def main():
    detector = dlib.get_frontal_face_detector()
    # CNN人脸检测器
    cnn_face_detector = \
        dlib.cnn_face_detection_model_v1(r'mmod_human_face_detector.dat')

    predictor_68_path = r"shape_predictor_68_face_landmarks.dat"

    predictor_68 = dlib.shape_predictor(predictor_68_path)

    frame_root_dir = r"E:\PycharmProjects\dataset\Oulu\Oulu_CASIA_NIR_VIS\VL\Strong"
    landmarks_root_dir = r'E:\PycharmProjects\dataset\Oulu\dlib_landmarks_0'

    dir_list = get_dir_list(frame_root_dir)
    i = 0
    t0 = time.time()

    for sub_dir in dir_list:
        frame_dir = frame_root_dir + "\\" + sub_dir
        landmarks_dir = landmarks_root_dir + "\\" + sub_dir
        mkdir_or_exist(landmarks_dir)
        total_img = []

        for _, _, files in os.walk(frame_dir):
            for data_file in files:
                if data_file.endswith(".jpeg"):
                    img = cv2.imread(frame_dir + '\\' + data_file)
                    idx = int(data_file.split(".")[0])
                    i = i + 1

                    face_rec = face_detector(img, detector, cnn_face_detector)
                    if face_rec is None:
                        print(sub_dir + '\\' + str(idx))
                        print("no face detected")
                        continue
                    landmarks_data = get_landmarks(img, face_rec, predictor_68)
                    if landmarks_data is None:
                        print(sub_dir + '\\' + str(idx))
                        print("no landmarks detected")
                        continue
                    txt_name = landmarks_dir + '\\' + str(idx) + "_my_landmarks.txt"
                    write_landmarks_txt(txt_name, landmarks_data)
                    if i >= 1000:
                        print("1000 images over")
                        print(sub_dir)
                        print(time.time() - t0)
                        i = 0
                        t0 = time.time()
                        print("\n")



if __name__=="__main__":
    main()
