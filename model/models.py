import torch

def create_model(opt):

    if opt.one_side:
        if opt.no_D:
            from .one_side_model import OneSideModel_wo_D
            model=OneSideModel_wo_D()
            model.initialize(opt)
            print ('Use OneSideModel_wo_D')
            print('model [%s] was created' % (model.name()))
        elif opt.use_vae:
            from .one_side_model_vae import OneSideModelVae
            model=OneSideModelVae()
            model.initialize(opt)
            print('model [%s] was created' % (model.name()))
        else:
            from .one_side_model import OneSideModel
            model=OneSideModel()
            model.initialize(opt)
            print('model [%s] was created' % (model.name()))
    else:
        if opt.use_final:
            from .mixture_model import MixtureModel
            model=MixtureModel()
            model.initialize(opt)
            print('model [%s] was created' % (model.name()))
        else:
            from .pix2pixHD_model import Pix2PixHDModel
            model = Pix2PixHDModel()
            model.initialize(opt)
            print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
