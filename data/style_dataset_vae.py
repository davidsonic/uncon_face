import os
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import glob
import numpy as np
import copy
import torch

class StyleDatasetVae(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.use_vae=opt.use_vae

        dirs = os.listdir(self.root)
        self.iden_nums = len(dirs)

        ori_files = []
        land_files = []
        idns = []

        for i in range(len(dirs)):
            # can be changed to original_face_discon
            ori_name = os.path.join(self.root, dirs[i], 'original_face')
            land_name = os.path.join(self.root, dirs[i], 'landmarks_face')

            ori_paths = glob.glob(os.path.join(ori_name, '*png'))
            land_paths = glob.glob(os.path.join(land_name, '*png'))

            ori_files.extend(ori_paths)
            land_files.extend(land_paths)

            idns.extend([dirs[i] for j in range(len(ori_paths))])

        self.ori_files = ori_files
        self.land_files = land_files
        self.dataset_size = len(self.ori_files)
        self.idns = np.array(idns)
        self.styles=['style_la_muse', 'style_rain', 'style_wave']


    # ori1 and land1 corresponds; or2 and land2 corresponds
    def __getitem__(self, index):
        idn = self.idns[index]
        choices = np.where(self.idns == idn)[0]
        choices = choices.tolist()
        choices.remove(index)
        idx = np.random.choice(choices)
        styles=copy.deepcopy(self.styles)

        ori1_path = self.ori_files[index]
        ori1 = Image.open(ori1_path).convert('RGB')
        params = get_params(self.opt, ori1.size)
        transform_image = get_transform(self.opt, params)
        ori1_tensor = transform_image(ori1)

        style=np.random.choice(styles)
        ori2_path = self.ori_files[idx].replace('original_face',style)
        ori2 = Image.open(ori2_path).convert('RGB')
        ori2_tensor = transform_image(ori2)

        land1_path = self.land_files[index]
        land1 = Image.open(land1_path).convert('L')
        params = get_params(self.opt, land1.size)
        transform_land = get_transform(self.opt, params)
        land1_tensor = transform_land(land1)

        # land2 should correspond to ori1
        land2_path = self.land_files[idx]
        land2 = Image.open(land2_path).convert('L')
        land2_tensor = transform_land(land2)


        direc1=torch.FloatTensor([0,0])

        if style=='style_la_muse':
            direc2=torch.FloatTensor([0,1])
        elif style=='style_rain':
            direc2=torch.FloatTensor([1,0])
        elif style=='style_wave':
            direc2=torch.FloatTensor([1,1])


        if self.use_vae:
            input_dict = {'land1': land1_tensor, 'ori1': ori1_tensor, 'land2': land2_tensor,
                         'ori2': ori2_tensor, 'direc1': direc1, 'direc2': direc2}
        else:
            input_dict = {'land1': land1_tensor, 'ori1': ori1_tensor, 'land2': land2_tensor,
                      'ori2': ori2_tensor}

        return input_dict

    def __len__(self):
        return min(self.dataset_size, self.opt.max_dataset_size)

    def name(self):
        return 'StyleDatasetVae'
