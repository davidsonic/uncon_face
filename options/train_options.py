import argparse
import os
from util import util
import torch

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):    
        # experiment specifics
        # self.parser.add_argument('--name', type=str, default='style_split_two_side_train_fix_flip_2_2_vgg_sim5_attention_augD_final', help='name of the experiment. It decides where to store samples and models')
        # self.parser.add_argument('--name', type=str, default='style2style_rafd_two_side_train_fix_flip_2_2_vgg_sim5', help='name of the experiment. It decides where to store samples and models')
        # self.parser.add_argument('--name', type=str, default='split_10_one_side_train_fix_flip_5xl1_vgg_pretrain_loss', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--name', type=str, default='photo_to_multi_style_fix_photo_5l1_vgg_5_2', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/mnt/jiali/code/uncon_face_checkpoint/checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')        
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')

        # input/output sizes
        # set false to use_encoding, which means transfer direction
        self.parser.add_argument('--use_encoding', action='store_false', help='use dropout for the generator')
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        # try one-hot encoding instead of direct input, original pic+68 one-hot channel map, add segmentation mask
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='/home/jiali/program/uncon_face/datasets/datasets/train')  # important!
        # self.parser.add_argument('--dataroot', type=str, default='/home/jiali/program/tmp/Rafd_category')
        # self.parser.add_argument('--dataroot', type=str, default='/mnt/jiali/data/icv/crop_resize')
        # self.parser.add_argument('--dataroot', type=str, default='/home/jiali/program/tmp/CUB_200_2011')  # important!
        self.parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        self.parser.add_argument('--no_flip', action='store_false', help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_false', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--n_downsample_global', type=int, default=3, help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=3, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--n_local_enhancers', type=int, default=0, help='number of local enhancers to use')
        self.parser.add_argument('--niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')        

        # for instance-wise features
        # landmarks all in the same channel
        self.parser.add_argument('--no_landmark', action='store_true', help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--label_feat', action='store_true', help='if specified, add encoded label features as input')        
        self.parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')        
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder') 
        self.parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')
        
        self.initialized = True

        # display
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true',
                                 help='only do one epoch and displays at each iteration')

        # for training
        # set false to continue train
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # for discriminators
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')


        #also used for vgg_loss weight, won't use in current version
        self.parser.add_argument('--lambda_feat', type=float, default=5.0, help='weight for feature matching loss')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true',
                                 help='if specified, do *not* use discriminator feature matching loss')

        #replace L1 loss with vgg loss to prevent overfitting, set true to use
        self.parser.add_argument('--no_vgg_loss', action='store_true',
                                 help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0,
                                 help='the size of image buffer that stores previously generated images')

        self.parser.add_argument('--lambda_A', type=float, default=5.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=5.0, help='weight for cycle loss (B -> A -> B)')
        # set true to use similarity loss
        self.parser.add_argument('--no_sim', action='store_false',
                                 help='if specified, do *not* add similarity loss')
        self.parser.add_argument('--lambda_sim', type=float, default=5.0, help='weight for similarity')

        self.isTrain = True

        # add L1, set this as false
        self.parser.add_argument('--add_L1', action='store_false', help='add L1 loss')
        self.parser.add_argument('--lambda_L1', type=float, default=5.0, help='weight for L1')

        # train two side model, set this as true
        self.parser.add_argument('--one_side', action='store_true', help='one side generation')

        self.parser.add_argument('--use_celeba', action='store_true', help='use celeba')

        # no D, later on, don't need this param
        self.parser.add_argument('--no_D', action='store_true', help='no D')

        # disentanglement of FG and BG
        self.parser.add_argument('--use_seg', action='store_true', help='feed segmentation input')
        self.parser.add_argument('--use_disen', action='store_true', help='disentangle foreground and background')

        # bird dataset
        self.parser.add_argument('--use_cub', action='store_true', help='train on CUB bird dataset')

        # style2style

        # no identity loss
        self.parser.add_argument('--identity', type=float, default=0.0, help='identity loss')
        # style and vae used together
        self.parser.add_argument('--use_style', action='store_false', help='with style transfer')
        self.parser.add_argument('--use_vae', action='store_false', help='use vae to generate different styles')
        # set false to use unpair_style
        self.parser.add_argument('--unpair_style', action='store_true', help='use vae to generate different styles')

        # use final_model?
        self.parser.add_argument('--use_final', action='store_true', help='use final version')


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk        
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt


if __name__=='__main__':
    opt=BaseOptions()
    opt.initialize()
    opt= opt.parse()