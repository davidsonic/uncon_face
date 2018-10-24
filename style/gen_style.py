import os
import sys

os.chdir('/home/jiali/program/pytorch_examples/fast_neural_style')
ROOT = '/home/jiali/program/uncon_face/datasets/datasets/train'
ids = os.listdir(ROOT)
count = 1
for id in ids:
    ori_path = os.path.join(ROOT, id, 'original_face')
    print 'current id is: {}, progress: {}'.format(id, count / len(ids))
    save_path = os.path.join(ROOT, id, 'style_mosaic')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    files = os.listdir(ori_path)
    for file in files:
        file_path = os.path.join(ori_path, file)
        save_file = os.path.join(save_path, file)
        # generate style
        cm='python neural_style/neural_style.py eval'
        ct_img=' --content-image '+file_path
        model=' --model '+'saved_models/mosaic.pth'
        out_img=' --output-image '+save_file
        cuda=' --cuda 3'
        cmd=cm+ct_img+model+out_img+cuda
        os.system(cmd)
