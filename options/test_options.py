import argparse
import os
import torch

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False


    def initialize(self):
        # experiment specifics
        # self.parser.add_argument('--name', type=str, default='split_10_one_side_train_fix_flip_5xl1_vgg', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--name', type=str, default='style_split_two_side_train_fix_flip_2_2_vgg_sim5', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/mnt/jiali/code/uncon_face_checkpoint/checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')

        # input/output sizes
        # set encoding as false to use
        self.parser.add_argument('--use_encoding', action='store_false', help='use dropout for the generator')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        # self.parser.add_argument('--dataroot', type=str, default='/home/jiali/program/uncon_face/datasets/online/')  # important!
        # self.parser.add_argument('--dataroot', type=str, default='/home/jiali/program/tmp/Rafd_category/71/')  # important!
        # self.parser.add_argument('--dataroot', type=str, default='/home/jiali/program/uncon_face/datasets/datasets/test/obama')  # important!
        self.parser.add_argument('--dataroot', type=str, default='/home/jiali/program/uncon_face/datasets/datasets/test/')  # important!
        # self.parser.add_argument('--dataroot', type=str, default='/home/jiali/program/uncon_face/no_D')  # important!
        self.parser.add_argument('--resize_or_crop', type=str, default='resize',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_false',
                                 help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
        self.parser.add_argument('--tf_log', action='store_false',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=4,
                                 help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=9,
                                 help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3,
                                 help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=0, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0,
                                 help='number of epochs that we only train the outmost local enhancer')

        # for instance-wise features
        self.parser.add_argument('--no_landmark', action='store_true',
                                 help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--instance_feat', action='store_true',
                                 help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true',
                                 help='if specified, add encoded label features as input')
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
        self.parser.add_argument('--load_features', action='store_true',
                                 help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

        self.initialized = True
        self.isTrain = False

        # self.parser.add_argument('--one_side', action='store_false', help='one side generation')
        self.parser.add_argument('--one_side', action='store_true', help='one side generation')

        # temporarily use celeba for testing generalization ability
        self.parser.add_argument('--use_celeba', action='store_true', help='use celeba')
        self.parser.add_argument('--results_dir', type=str, default='./results2/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',  # latest_finetune
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=500, help='how many test images to run')
        self.parser.add_argument('--no_D', action='store_true', help='no D')

        self.parser.add_argument('--use_cub', action='store_true', help='train on CUB bird dataset')

        # style 2 style
        self.parser.add_argument('--use_style', action='store_true', help='with style transfer')
        #
        self.parser.add_argument('--unpair_style', action='store_true', help='use vae to generate different styles')
        self.parser.add_argument('--use_fashion', action='store_true', help='use deep fashion dataset')
        self.parser.add_argument('--use_seg', action='store_true', help='feed segmentation input')
        self.parser.add_argument('--use_vae', action='store_true', help='use vae to generate different styles')
        self.parser.add_argument('--use_final', action='store_true', help='use final version')
        #
        # use final_model?
        # self.parser.add_argument('--use_final', action='store_true', help='use final version')


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt=self.parser.parse_args()
        self.opt.isTrain=self.isTrain

        str_ids=self.opt.gpu_ids.split(',')
        self.opt.gpu_ids=[]
        for str_id in str_ids:
            id=int(str_id)
            if id>=0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids)>0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args=vars(self.opt)

        print '-'*10+' Options '+'-'*10
        for k, v in sorted(args.items()):
            print ('%s: %s' %(str(k), str(v)))
        print '-'*10+' End '+'-'*10

        return self.opt



if __name__=='__main__':
    opts=TestOptions().parse()


