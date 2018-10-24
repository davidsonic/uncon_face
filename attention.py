# Test for only one picture
# Note, attention requires changing networks.py's globalGenerator2

from options.attention_options import AttenOptions
from model.models import create_model
from data.base_dataset import get_transform, get_params
from PIL import Image
from torch.autograd import Variable
import cv2
import os
import numpy as np
from util.util import tensor2im, save_image
import time
from torch.nn import functional as F
import pdb


def init_options():
    opt=AttenOptions()
    opt=opt.parse()
    opt.nThreads=1
    opt.batchSize=1
    opt.serial_batches=False
    opt.no_flip=True
    return opt

def prepare_data(opt,filename):
    # generate data
    ROOT=os.path.join(opt.dataroot,'web')
    ori_image=os.path.join(ROOT,'original_face',filename)
    land_image=os.path.join(ROOT,'landmarks_face',filename)

    ori=Image.open(ori_image).convert('RGB')
    land=Image.open(land_image).convert('L')

    params=get_params(opt,ori.size)
    transform_image=get_transform(opt,params)

    ori_tensor=transform_image(ori)
    land_tensor=transform_image(land)

    ori_tensor=ori_tensor.unsqueeze(0)
    land_tensor=land_tensor.unsqueeze(0)
    return ori_tensor, land_tensor

def init_model(opt):
    model=create_model(opt)
    return model

def inference(opt,model,ori_tensor, land_tensor):
    generated=model.inference(Variable(land_tensor), Variable(ori_tensor))
    newname=str(hash(time.time()))+'.png'
    # import pdb
    # pdb.set_trace()
    img_np=tensor2im(generated.data[0])
    save_image(img_np,os.path.join(opt.results_dir,newname))
    return newname


def inference2(opt,model,ori_tensor, land_tensor):
    generated=model.inference(Variable(land_tensor), Variable(ori_tensor))
    img_np=tensor2im(generated.data[0])
    save_image(img_np, os.path.join(os.getcwd(),'test.png'))



def unit_test():
    opt = init_options()
    model = init_model(opt)
    ori_tensor, land_tensor = prepare_data(opt)
    inference(opt, model, ori_tensor, land_tensor)


# debug
def prepare_data2(opt):
    # generate data
    ROOT=os.path.join(opt.dataroot,'web')
    ori_paths=os.path.join(ROOT,'original_face')
    land_paths=os.path.join(ROOT,'landmarks_face')
    ori_files=os.listdir(ori_paths)
    land_files=os.listdir(land_paths)

    ori_tensors=[]
    land_tensors=[]
    for ori in ori_files:
        ori_image=os.path.join(ori_paths,ori)
        land_image=os.path.join(land_paths,ori)

        ori=Image.open(ori_image).convert('RGB')
        land=Image.open(land_image).convert('L')

        params=get_params(opt,ori.size)
        transform_image=get_transform(opt,params)

        ori_tensor=transform_image(ori)
        land_tensor=transform_image(land)

        ori_tensor=ori_tensor.unsqueeze(0)
        land_tensor=land_tensor.unsqueeze(0)

        ori_tensors.append(ori_tensor)
        land_tensors.append(land_tensor)
    return ori_tensors, land_tensors


def unit_test2():
    opt = init_options()
    model = init_model(opt)
    ori_tensors, land_tensors = prepare_data2(opt)

    for (ori_tensor, land_tensor) in zip(ori_tensors, land_tensors):
        inference(opt, model, ori_tensor, land_tensor)


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


# hook the feature extractor
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())




if __name__=='__main__':
    path='/home/jiali/program/uncon_face/paper'
    ori=os.path.join(path,'1.png')
    land=os.path.join(path,'land2.png')
    opt = init_options()

    # prepare data
    ori = Image.open(ori).convert('RGB')
    land = Image.open(land).convert('L')

    params = get_params(opt, ori.size)
    transform_image = get_transform(opt, params)

    ori_tensor = transform_image(ori)
    land_tensor = transform_image(land)

    ori_tensor = ori_tensor.unsqueeze(0)
    land_tensor = land_tensor.unsqueeze(0)

    # generate
    model=init_model(opt)
    model.eval()
    # get the softmax weight
    params = list(model.parameters())
    features_blobs = []
    conv_name='netG'
    # print model._modules
    # pdb.set_trace()
    model._modules.get(conv_name).register_forward_hook(hook_feature)
    inference2(opt,model, ori_tensor, land_tensor)
    print 'img done!'












