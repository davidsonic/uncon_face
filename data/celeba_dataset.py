
import os
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import glob
import numpy as np
from experiment.iden_stats import get_train_test_ids

class CelebaDataset(BaseDataset):
    def initialize(self, opt):

        self.opt = opt
        self.root =opt.dataroot

        train_dic, test_dic = get_train_test_ids()
        self.iden_nums = len(train_dic.keys())

        ori_files = []
        land_files = []
        idns = []

        for k,v in train_dic.iteritems():
            ori_name=os.path.join(self.root, k, 'original_face')
            land_name=os.path.join(self.root, k, 'landmarks_face')

            ori_paths=glob.glob(os.path.join(ori_name, '*jpg'))
            land_paths=glob.glob(os.path.join(land_name, '*jpg'))


            ori_files.extend(ori_paths)
            land_files.extend(land_paths)
            idns.extend([k for j in range(len(v))])


        self.ori_files=ori_files
        self.land_files=land_files
        self.dataset_size = len(self.ori_files)
        self.idns= np.array(idns)

    # ori1 and land1 corresponds; or2 and land2 corresponds
    def __getitem__(self, index):        
        idn = self.idns[index]
        choices=np.where(self.idns==idn)[0]
        choices=choices.tolist()
        choices.remove(index)
        idx=np.random.choice(choices)

        ori1_path = self.ori_files[index]
        ori1 = Image.open(ori1_path).convert('RGB')
        params = get_params(self.opt, ori1.size)
        transform_image = get_transform(self.opt, params)
        ori1_tensor = transform_image(ori1)

        ori2_path = self.ori_files[idx]
        ori2=Image.open(ori2_path).convert('RGB')
        ori2_tensor=transform_image(ori2)


        land1_path=self.land_files[index]
        land1=Image.open(land1_path).convert('L')
        params=get_params(self.opt, land1.size)
        transform_land=get_transform(self.opt, params)
        land1_tensor=transform_land(land1)

        # land2 should correspond to ori1
        land2_path=self.land_files[idx]
        land2=Image.open(land2_path).convert('L')
        land2_tensor=transform_land(land2)


        input_dict = {'land1': land1_tensor, 'ori1': ori1_tensor, 'land2': land2_tensor,
                      'ori2': ori2_tensor}

        return input_dict

    def __len__(self):
        return min(self.dataset_size, self.opt.max_dataset_size)

    def name(self):
        return 'CelebaDataset'