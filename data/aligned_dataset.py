
import os
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import glob
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        dirs=os.listdir(self.root)
        self.iden_nums = len(dirs)


        ori_files = []
        land_files = []
        seg_files=[]
        idns = []

        # print dirs
        for i in range(len(dirs)):
            # can be changed to original_face_discon
            ori_name=os.path.join(self.root, dirs[i], 'original_face')
            land_name=os.path.join(self.root, dirs[i], 'landmarks_face')
        # land_name=os.path.join(self.root, dirs[i], 'landmarks')
        # seg_name =os.path.join(self.root, dirs[i], 'segmentation')

        # ori_paths=glob.glob(os.path.join(ori_name, '*png'))
            ori_paths=glob.glob(os.path.join(ori_name, '*'))
        # land_paths=glob.glob(os.path.join(land_name, '*png'))
            land_paths=glob.glob(os.path.join(land_name, '*'))

            if opt.use_seg:
                seg_paths=glob.glob(os.path.join(seg_name,'*png'))
                seg_files.extend(seg_paths)

            ori_files.extend(ori_paths)
            land_files.extend(land_paths)

            idns.extend([dirs[i] for j in range(len(ori_paths))])
        # idns.extend(['obama' for j in range(len(ori_paths))])


        self.ori_files=ori_files
        self.land_files=land_files
        if opt.use_seg:
            self.seg_files=seg_files
        self.dataset_size = len(self.ori_files)
        print 'AlignedDataset size: {}'.format(self.dataset_size)
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

        # segmentation input
        if self.opt.use_seg:
            transform_seg1 = get_transform(self.opt, params)
            transform_seg2 = get_transform(self.opt, params, normalize=False)

            seg1_path=self.seg_files[index]
            seg1=Image.open(seg1_path).convert('L')
            seg1_ori=transform_seg2(seg1)
            # input must be normalized to same range, don't need 1 or 0 ?
            seg1_tensor= transform_seg1(seg1)


            seg2_path=self.seg_files[idx]
            seg2=Image.open(seg2_path).convert('L')
            seg2_ori=transform_seg2(seg2)
            # In order to disentangle, don't normalize && 1, 0 value
            # seg2_tensor=self.seg2mask(seg2, transform_seg2)
            # try flipseg
            seg2_tensor=transform_seg1(seg2)


        if self.opt.use_seg:
            input_dict = {'land1': land1_tensor, 'ori1': ori1_tensor, 'seg1': seg1_tensor, 'seg1_ori': seg1_ori,  'land2': land2_tensor,
                          'ori2': ori2_tensor, 'seg2': seg2_tensor, 'seg2_ori':seg2_ori}
        else:
            input_dict = {'land1': land1_tensor, 'ori1': ori1_tensor, 'land2': land2_tensor,
                      'ori2': ori2_tensor}

        return input_dict


    def seg2mask(self, seg, transform):

        seg_np=np.array(seg)
        seg_np[seg_np==255]=1
        seg_np[seg_np==0]=0
        seg_PIL=Image.fromarray(seg_np)
        seg_tensor=transform(seg_PIL)
        return  seg_tensor


    def __len__(self):
        return min(self.dataset_size, self.opt.max_dataset_size)

    def name(self):
        return 'AlignedDataset'