import os
from shutil import copy2
import glob
import numpy as np

ratio=0.1
ROOT='/home/jiali/program/uncon_face/datasets/datasets/origin'
SAVE_TRAIN='/home/jiali/program/uncon_face/datasets/datasets/train'
SAVE_TEST='/home/jiali/program/uncon_face/datasets/datasets/test'
dirs=os.listdir(ROOT)
# can be changed to original_face_discon
ori_dir='original_face'
land_dir='landmarks_face'


for i in range(len(dirs)):
    dir_path=os.path.join(ROOT, dirs[i])
    dir_ori_files=glob.glob(os.path.join(dir_path,ori_dir, '*.png'))
    dir_land_files=glob.glob(os.path.join(dir_path, land_dir, '*.png'))

    test_nums=int(len(dir_ori_files)*ratio)


    dir_test_idx=np.random.choice(len(dir_ori_files),test_nums)
    dir_train_idx=list(set(range(len(dir_ori_files))) - set(dir_test_idx))


    dir_train_ori_files=np.array(dir_ori_files)[dir_train_idx]
    dir_train_land_files=np.array(dir_land_files)[dir_train_idx]
    dir_test_ori_files=np.array(dir_ori_files)[dir_test_idx]
    dir_test_land_files=np.array(dir_land_files)[dir_test_idx]


    dest_train_ori_path=os.path.join(SAVE_TRAIN, dirs[i], ori_dir)
    if not os.path.isdir(dest_train_ori_path):
        os.makedirs(dest_train_ori_path)
    dest_train_land_path=os.path.join(SAVE_TRAIN, dirs[i], land_dir)
    if not os.path.isdir(dest_train_land_path):
        os.makedirs(dest_train_land_path)
    dest_test_ori_path=os.path.join(SAVE_TEST, dirs[i], ori_dir)
    if not os.path.isdir(dest_test_ori_path):
        os.makedirs(dest_test_ori_path)
    dest_test_land_path=os.path.join(SAVE_TEST, dirs[i], land_dir)
    if not os.path.isdir(dest_test_land_path):
        os.makedirs(dest_test_land_path)

    for file in dir_train_ori_files:
        copy2(file, dest_train_ori_path)

    for file in dir_train_land_files:
        copy2(file, dest_train_land_path)

    for file in dir_test_ori_files:
        copy2(file, dest_test_ori_path)


    for file in dir_test_land_files:
        copy2(file, dest_test_land_path)










