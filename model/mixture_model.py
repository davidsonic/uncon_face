
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import pdb

#debug
# from base_options import BaseOptions
# from base_model import BaseModel
# import networks

class MixtureModel(BaseModel):
    def name(self):
        return 'MixtureModel'


    def similarity_loss(self, input, output):
        error=torch.mean(torch.abs(output-input))
        return error


    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc

        ##### define networks        
        # Generator network
        #input_nc is 3
        netG_input_nc = input_nc

        if not opt.no_landmark:
            netG_input_nc += 1
        if opt.use_encoding:
            netG_input_nc += 1
        if opt.netG!='mixture':
            self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
        else:
            self.netG1, self.netG2=networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # plus target landmark as condition
            netD_input_nc = input_nc + 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        # util function
        self.downsample_img = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  


        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.real_pool=ImagePool(opt.pool_size)
            self.land_pool=ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            self.loss_names = ['loss_G_A', 'loss_G_B', 'loss_cycle_A', 'loss_cycle_B', 'loss_D']
            # Names so we can breakout loss
            if self.opt.identity>0:
                self.loss_names.extend(['loss_idt_A', 'loss_idt_B'])

            if not self.opt.no_sim:
                self.loss_names.extend(['loss_sim_A', 'loss_sim_B'])

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]
            else:
                params = list(self.netG.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D                        
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))


    def encode_input(self, land, ori, direc=1.0):

        encoding = torch.cuda.FloatTensor(torch.Size(land.size())).fill_(direc)
        land=land.data.cuda()
        ori=ori.data.cuda()

        input=torch.cat((ori,land,encoding),dim=1)
        return Variable(input)



    def forward(self, land1, ori1, land2, ori2):
        # Encode Inputs

        self.input = self.encode_input(land2, ori1, direc=1.0)
        self.output = self.encode_input(land1, ori2, direc=0.0)

        self.land1=Variable(land1.data.cuda())
        self.ori1=Variable(ori1.data.cuda())
        self.land2=Variable(land2.data.cuda())
        self.ori2=Variable(ori2.data.cuda())

        # extra dimension of landmarks
        self.land2_d2=self.downsample_img(self.land2)
        self.land2_d4=self.downsample_img(self.land2_d2)

        self.land1_d2=self.downsample_img(self.land1)
        self.land1_d4=self.downsample_img(self.land1_d2)

        # for visualization, but don't meed actually
        # self.land1 = land1.data
        # self.ori1 = ori1.data
        # self.land2 = land2.data
        # self.ori2 = ori2.data


    def backward_G(self):

        lambda_idt=self.opt.identity
        if lambda_idt>0:
            input=self.encode_input(self.land2, self.ori2, direc=1.0)
            idt_A=self.netG(input)
            loss_idt_A=self.criterionIdt(idt_A, self.ori2) * self.opt.lambda_B * lambda_idt
            output=self.encode_input(self.land1, self.ori1, direc=0.0)
            idt_B=self.netG(output)
            loss_idt_B=self.criterionIdt(idt_B, self.ori1) * self.opt.lambda_A * lambda_idt

        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_sim = self.opt.lambda_sim

        pyrd_fake_B=self.netG(self.input)
        fake_augs=[]
        fake_augs.append(torch.cat((pyrd_fake_B[0],self.land2_d4),dim=1))
        fake_augs.append(torch.cat((pyrd_fake_B[1],self.land2_d2),dim=1))
        fake_augs.append(torch.cat((pyrd_fake_B[2],self.land2),dim=1))
        # add target landmark as condition input
        # fake_B_aug=torch.cat((fake_B, self.land2), dim=1)
        pred_fake=self.netD(fake_augs)
        loss_G_A=self.criterionGAN(pred_fake, True)

        pyrd_fake_A=self.netG(self.output)
        # aug fake_A with landmark input
        # fake_A_aug=torch.cat((fake_A, self.land1), dim=1)
        fake_augs = []
        fake_augs.append(torch.cat((pyrd_fake_A[0],self.land1_d4), dim=1))
        fake_augs.append(torch.cat((pyrd_fake_A[1],self.land1_d2), dim=1))
        fake_augs.append(torch.cat((pyrd_fake_A[2],self.land1), dim=1))
        pred_fake=self.netD(fake_augs)
        loss_G_B=self.criterionGAN(pred_fake, True)

        # similarity loss
        if not self.opt.no_sim:
            loss_sim_A=self.similarity_loss(self.ori2, pyrd_fake_B[2]) * lambda_sim
            loss_sim_B=self.similarity_loss(self.ori1, pyrd_fake_A[2]) * lambda_sim


        # actually don't need two directions?
        fake_B_in=self.encode_input(self.land1, pyrd_fake_B[2], direc=0.0)
        pyrd_rec_A=self.netG(fake_B_in)
        rec_A=pyrd_rec_A[2]
        loss_cycle_A=self.criterionCycle(rec_A, self.ori1) * lambda_A

        fake_A_in=self.encode_input(self.land2, pyrd_fake_A[2], direc=1.0)
        pyrd_rec_B=self.netG(fake_A_in)
        rec_B=pyrd_rec_B[2]
        loss_cycle_B=self.criterionCycle(rec_B, self.ori2) * lambda_B

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B
        if not self.opt.no_sim:
            loss_G += loss_sim_A + loss_sim_B

        if lambda_idt>0:
            loss_G+=loss_idt_A + loss_idt_B


        loss_G.backward()


        self.fake_B =pyrd_fake_B[2].data
        self.fake_A=pyrd_fake_A[2].data
        self.rec_A= rec_A.data
        self.rec_B=rec_B.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]

        if lambda_idt>0:
            self.loss_idt_A=loss_idt_A.data[0]
            self.loss_idt_B=loss_idt_B.data[0]

        if not self.opt.no_sim:
            self.loss_sim_A=loss_sim_A.data[0]
            self.loss_sim_B=loss_sim_B.data[0]


    def backward_D_basic(self, netD, real, fake, land):
        # Real
        # aug landmark input
        real_aug=torch.cat((real, land), dim=1)
        pred_real = netD(real_aug)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        #aug landmark input
        fake_aug=torch.cat((fake, land), dim=1)
        pred_fake = netD(fake_aug.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        idx=np.random.randint(2)
        if idx==1:
            # fake_A corresponds to land1
            fake = self.fake_pool.query(Variable(self.fake_A))
            real = self.real_pool.query(self.ori1)
            land = self.land_pool.query(self.land1)
        else:
            fake=self.fake_pool.query(Variable(self.fake_B))
            real=self.real_pool.query(self.ori2)
            land =self.land_pool.query(self.land2)

        loss_D = self.backward_D_basic(self.netD, real, fake, land)
        self.loss_D=loss_D.data[0]


    def optimize_parameters(self):
        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()


        losses=[self.loss_G_A, self.loss_G_B, self.loss_cycle_A, self.loss_cycle_B, self.loss_D]
        if self.opt.identity>0:
            losses.extend([self.loss_idt_A,self.loss_idt_B])

        if not self.opt.no_sim:
            losses.extend([self.loss_sim_A, self.loss_sim_B])

        fake_images={'fake_A': self.fake_A, 'fake_B': self.fake_B,
                     'rec_A': self.rec_A, 'rec_B': self.rec_B}

        return losses, fake_images


    def inference(self, label, inst):
        # Encode Inputs        
        input_label, inst_map, _, _ = self.encode_input(Variable(label), Variable(inst), infer=True)

        # Fake Generation
        if self.use_features:       
            # sample clusters from precomputed features             
            feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label                
        fake_image = self.netG.forward(input_concat)
        return fake_image


    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)


    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr



if __name__=='__main__':
    opt = BaseOptions()
    opt.initialize()
    opt = opt.parse()

    model=Pix2PixHDModel()
    model.initialize(opt)
