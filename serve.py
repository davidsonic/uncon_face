# Test for only one picture
from options.serve_options import ServerOptions
from model.models import create_model
from data.base_dataset import get_transform, get_params
from PIL import Image
from torch.autograd import Variable
import cv2
import os
import numpy as np
from util.util import tensor2im, save_image
import time


def init_options():
    opt=ServerOptions()
    opt=opt.parse()
    opt.nThreads=1
    opt.batchSize=1
    opt.serial_batches=False
    opt.no_flip=True
    return opt

def prepare_data(opt,filename,direction=0.0):
    # generate data
    ROOT=os.path.join(opt.dataroot,'web')
    if(direction==0.0):
        ori_image=os.path.join(ROOT,'original_face',filename)
    else:
        ori_image = os.path.join(ROOT, 'results', filename)
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

def inference(opt,model,ori_tensor, land_tensor, direc=0.0):
    # generated=model.inference(Variable(land_tensor), Variable(ori_tensor), direc=direc)
    generated=model.inference(Variable(land_tensor), Variable(ori_tensor))
    newname=str(hash(time.time()))+'.png'
    # import pdb
    # pdb.set_trace()
    img_np=tensor2im(generated.data[0])
    save_image(img_np,os.path.join(opt.results_dir,newname))
    return newname



def unit_test():
    opt = init_options()
    model = init_model(opt)
    ori_tensor, land_tensor = prepare_data(opt, '2967942238.png')
    inference(opt, model, ori_tensor, land_tensor)


# debug
def prepare_data2(opt):
    # generate data
    ROOT=os.path.join(opt.dataroot,'web')
    ori_paths=os.path.join(ROOT,'original_face')
    land_paths=os.path.join(ROOT,'landmarks_face')
    ori_files=os.listdir(ori_paths)
    land_files=os.listdir(land_paths)
    print 'len files: {}'.format(len(land_paths))

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

    res=[]
    for (ori_tensor, land_tensor) in zip(ori_tensors, land_tensors):
        generated=inference(opt, model, ori_tensor, land_tensor)



if __name__=='__main__':
    # path='/home/jiali/program/uncon_face/no_D'
    # ori=os.path.join(path,'3.jpg')
    # land=os.path.join(path,'3_land.jpg')
    # opt = init_options()
    #
    # # prepare data
    # ori = Image.open(ori).convert('RGB')
    # land = Image.open(land).convert('L')
    #
    # params = get_params(opt, ori.size)
    # transform_image = get_transform(opt, params)
    #
    # ori_tensor = transform_image(ori)
    # land_tensor = transform_image(land)
    #
    # ori_tensor = ori_tensor.unsqueeze(0)
    # land_tensor = land_tensor.unsqueeze(0)
    #
    # # generate
    # model=init_model(opt)
    # gen_img=inference(opt,model,ori_tensor,land_tensor)
    # print 'img generated'

    ###### unit test of two side model #####
    unit_test()








