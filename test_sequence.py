import os
from collections import OrderedDict
from data.data_loader import CreateDataLoader
from model.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from options.test_options import TestOptions
from torch.autograd import Variable
import pdb
from gen_landmark import gen_landmark, init, get_land_image, interpolate_land_image
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
import cv2
import torch


opt = TestOptions().parse()
opt.nThreads=1
opt.batchSize=1
opt.serial_batches=False # no shuffle
opt.no_flip=True

model=create_model(opt)
visualizer=Visualizer(opt)

BASE_DIR='/home/jiali/program/uncon_face/datasets/datasets/test/obama'
ori_path=os.path.join(BASE_DIR,'original_face')
land_path=os.path.join(BASE_DIR,'landmarks_face')

ori_filename=os.path.join(ori_path,'Address_Obama_6.mp4-22.png')
ori1=Image.open(ori_filename).convert('RGB')
ori2=Image.open(ori_filename).convert('RGB')

land_filename=os.path.join(land_path,'Address_Obama_6.mp4-22.png')
land1=Image.open(land_filename).convert('L')

# get transform
params=get_params(opt, ori1.size)
transform=get_transform(opt, params)
ori1_tensor=torch.unsqueeze(transform(ori1),0)
ori2_tensor=torch.unsqueeze(transform(ori2),0)
land1_tensor=torch.unsqueeze(transform(land1),0)

iters=os.listdir(land_path)
for file in iters:
    filename=os.path.join(land_path,file)
    land2=Image.open(filename).convert('L')
    land2_tensor = torch.unsqueeze(transform(land2), 0)
    data = {'land1': land1_tensor, 'ori1': ori1_tensor, 'land2': land2_tensor, 'ori2': ori2_tensor}
    generated = model.inference(Variable(data['land2']), Variable(data['ori1']))
    save_path = os.path.join('paper', 'sequence',file)
    util.save_image(util.tensor2im(generated.data[0]), save_path)
