import os
import shutil

BASE_DIR='/home/jiali/program/uncon_face/datasets/datasets/test'
# DEST_DIR='/home/jiali/program/uncon_face/results2/two_style_nature2style_groundtruth'
DEST_DIR='/home/jiali/program/uncon_face/results2/wacv_test_input'

dirs=os.listdir(BASE_DIR)

i=0
for direc in dirs:
    # obama, hillary ...
    direc_path=os.path.join(BASE_DIR,direc)
    # subdir_path = os.path.join(direc_path, 'style_rain_princess')
    subdir_path = os.path.join(direc_path, 'original_face')
    direc_files=os.listdir(subdir_path)
    for file in direc_files:

        file_path=os.path.join(subdir_path,file)
        dest_path=os.path.join(DEST_DIR, '%05d.png' % i)
        print file_path
        shutil.copyfile(file_path, dest_path)
        i+=1
