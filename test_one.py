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

# prepare data
detector, predictor, DOWNSAMPLE_RATIO=init()
pic1=os.path.join('paper','rafd_exp1.jpg')
pic2=os.path.join('paper','rafd_exp2.jpg')

land1_img, land1=gen_landmark(pic1, detector, predictor, DOWNSAMPLE_RATIO)
cv2.imwrite(os.path.join('paper','rafd_land1.jpg'), land1_img)
land2_img, land2=gen_landmark(pic2, detector, predictor, DOWNSAMPLE_RATIO)
cv2.imwrite(os.path.join('paper','rafd_land2.jpg'), land2_img)

interp_land=interpolate_land_image(land1, land2)
interp_land_path=os.path.join('paper','rafd_interp_land.jpg')
cv2.imwrite(interp_land_path, interp_land)


ori1=Image.open(pic1).convert('RGB')
land1=Image.open(interp_land_path).convert('L')
ori2=Image.open(pic2).convert('RGB')
land2=Image.open(interp_land_path).convert('L')

# get transform
params=get_params(opt, ori1.size)
transform=get_transform(opt, params)
# get tensor
ori1_tensor=torch.unsqueeze(transform(ori1),0)
ori2_tensor=torch.unsqueeze(transform(ori2),0)
land1_tensor=torch.unsqueeze(transform(land1),0)
land2_tensor=torch.unsqueeze(transform(land2),0)


print ori1_tensor.size()
print land1_tensor.size()
print ori2_tensor.size()
print land2_tensor.size()

data={'land1': land1_tensor, 'ori1': ori1_tensor, 'land2':land2_tensor, 'ori2': ori2_tensor}
model=create_model(opt)
visualizer=Visualizer(opt)

generated=model.inference(Variable(data['land2']), Variable(data['ori1']))
save_path=os.path.join('paper','interp_pic.jpg')
util.save_image(util.tensor2im(generated.data[0]), save_path)

