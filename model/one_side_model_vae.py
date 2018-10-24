import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


# debug
# from base_options import BaseOptions
# from base_model import BaseModel
# import networks

class OneSideModelVae(BaseModel):
    def name(self):
        return 'OneSideModelVae'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none':  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc

        ##### define networks
        # Generator network
        # input_nc is 3
        netG_input_nc = input_nc

        if not opt.no_landmark:
            netG_input_nc += 1
        if opt.use_encoding:
            netG_input_nc += 1
        if opt.use_seg:
            netG_input_nc+=1
        # encoded feature as additional channel
        if opt.use_vae:
            netG_input_nc+=1
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        # use vae to encode features, input is 3, output is 1
        if self.opt.use_vae:
            self.netE=networks.define_E(opt.output_nc, opt.feat_num, opt.nef, opt.n_downsample_E)

        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            print 'pretrain_path: {}'.format(pretrained_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.use_vae:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)


                # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.real_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionFeat =torch.nn.L1Loss()

            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = ['loss_G_A', 'loss_D', 'G_VGG']
            if self.opt.add_L1:
                self.loss_names +=['loss_L1']
            if self.opt.use_vae:
                self.loss_names+=['loss_vae_kl']




# initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print(
                    '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params': [value], 'lr': opt.lr}]
                    else:
                        params += [{'params': [value], 'lr': 0.0}]
            else:
                params = list(self.netG.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            #optimizer for VAE netE
            if opt.use_vae:
                params=list(self.netE.parameters())
                self.optimizer_E=torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, land, ori, direc=1.0, seg=None, vae=None):

        land = land.data.cuda()
        ori = ori.data.cuda()


        input=torch.cat((ori,land), dim=1)
        if self.opt.use_seg:
            segin = seg.data.cuda()
            input=torch.cat((input, segin), dim=1)

        # if use vae, then vae as additional input
        if self.opt.use_vae:
            input=torch.cat((input, vae.data.cuda()),dim=1)

        return Variable(input)


    def forward(self, land1, ori1, land2, ori2, seg1=None, seg2=None):

        self.land1 = Variable(land1.data.cuda())
        self.ori1 = Variable(ori1.data.cuda())
        self.land2 = Variable(land2.data.cuda())
        self.ori2 = Variable(ori2.data.cuda())

        # Encode Inputs, input is ori2
        if self.opt.use_vae:
            self.mu, self.logvar = self.netE.forward(self.ori2)
            std=self.logvar.mul(0.5).exp_()
            eps=self.get_z_random(std.size(0), std.size(1), 'gauss')
            z_encoded=eps.mul(std).add_(self.mu)
            # replicate to the same size of ori1 and land
            self.z_encoded=z_encoded.expand(1,1,512,512)
        # print 'after z_encoded size: {}'.format(self.z_encoded.size())

        # don't use seg
        if self.opt.use_vae:
            self.input=self.encode_input(land2, ori1, vae=self.z_encoded)
        else:
            self.input=self.encode_input(land2, ori1)

        # if flipseg:
        #     self.input=self.encode_input(land2, ori1, seg=seg2)
        # else:
        #     self.input=self.encode_input(land2, ori1, seg=seg1)

        # disentangle fg and bg
        if self.opt.use_seg:
            self.seg1=Variable(seg1.data.cuda())
            self.seg2=Variable(seg2.data.cuda())

    def backward_G(self):

        fake_B = self.netG(self.input)
        pred_fake = self.netD(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        if self.opt.add_L1:
            if self.opt.use_disen:
                # if no constraint on bg, what will happen? If more emphasis is on the fg, then lambda_l1 cannot be too large, otherwise blurry result
                loss_L1=torch.sum(torch.abs((fake_B-self.ori2)*(1+self.seg2)))
            else:
                loss_L1 =self.criterionL1(fake_B, self.ori2)
            loss_G = loss_G_A + loss_L1 * self.opt.lambda_L1
            self.loss_L1 = loss_L1.data[0]
        else:
            loss_G = loss_G_A
        #
        # # GAN feature matching loss
        # loss_G_GAN_Feat = 0
        # if not self.opt.no_ganFeat_loss:
        #     feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        #     D_weights = 1.0 / self.opt.num_D
        #     for i in range(self.opt.num_D):
        #         for j in range(len(pred_fake[i]) - 1):
        #             loss_G_GAN_Feat += D_weights * feat_weights * \
        #                                self.criterionFeat(pred_fake[i][j],
        #                                                   pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG perceptual loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_B, self.ori2) * self.opt.lambda_feat

        loss_G+=loss_G_VGG

        # vae loss
        if self.opt.lambda_kl>0.0:
            kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            loss_kl=torch.sum(kl_element).mul_(-0.5)*self.opt.lambda_kl
            loss_G+=loss_kl


        loss_G.backward()

        self.loss_G_VGG = loss_G_VGG.data[0]
        # self.loss_G_GAN_Feat =loss_G_GAN_Feat.data[0]
        self.fake_B = fake_B.data
        self.loss_G_A = loss_G_A.data[0]

        if self.opt.use_vae:
            self.loss_kl=loss_kl.data[0]


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake = self.fake_pool.query(Variable(self.fake_B))

        # bug!
        # idx = np.random.randint(2)
        # if idx == 1:
        #     real = self.real_pool.query(self.ori1)
        # else:
        #     real = self.real_pool.query(self.ori2)

        real=self.real_pool.query(self.ori2)
        #
        loss_D = self.backward_D_basic(self.netD, real, fake)
        self.loss_D = loss_D.data[0]

    def optimize_parameters(self):
        # G, add E optimization
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_E.step()
        # D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        losses=[self.loss_G_A, self.loss_D, self.loss_G_VGG]

        if self.opt.add_L1:
            losses += [self.loss_L1]
        if self.opt.use_vae:
            losses+= [self.loss_kl]

        fake_images = {'fake_B': self.fake_B}

        return losses, fake_images

    # inference
    def inference(self, land2, ori1):
        input = self.encode_input(land2, ori1)

        fake_image= self.netG.forward(input)
        return fake_image


    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.opt.use_vae:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

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
        # update lr for netE
        for param_group in self.optimizer_E.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


    # generate random noize
    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z=self.Tensor(batchSize, nz)
        if random_type=='uni':
            z.copy_(torch.rand(batchSize, nz)*2.0-1.0)
        elif random_type=='gauss':
            z.copy_(torch.randn(batchSize, nz))
        z=Variable(z)
        return  z



if __name__ == '__main__':
    opt = BaseOptions()
    opt.initialize()
    opt = opt.parse()

    model = Pix2PixHDModel()
    model.initialize(opt)
