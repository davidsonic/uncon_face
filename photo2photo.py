import time
from collections import OrderedDict
# from options.fashion_options import FashionOptions
from options.base_options import BaseOptions
from data.data_loader import CreateDataLoader
from model.models import create_model

import torch
from torch.autograd import Variable
import numpy as np
import os

import util.util as util
from util.visualizer import Visualizer


# opt = FashionOptions()
opt = BaseOptions()
opt.initialize()
opt = opt.parse()

opt.tf_log=True
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0


opt.debug=False
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 5
    opt.niter_decay = 0
    opt.max_dataset_size = 10



dataLoader = CreateDataLoader(opt)
dataloader = dataLoader.load_data()
dataset_size=len(dataloader)
print 'dataset size: {}'.format(dataset_size)

#######
# view test_data result as well
# opt.dataroot='/home/jiali/program/face_LM/data/datasets/test'
# opt.how_many=1
# test_dataLoader = CreateDataLoader(opt)
# test_dataloader = test_dataLoader.load_data()
# test_dataset_size=len(test_dataloader)
# print 'test dataset size: {}'.format(test_dataset_size)



# set model
model=create_model(opt)
visualizer=Visualizer(opt)

# set training
total_steps = (start_epoch-1) * dataset_size + epoch_iter

for epoch in range(start_epoch, opt.niter+opt.niter_decay+1):
    epoch_start_time=time.time()
    if epoch!=start_epoch:
        epoch_iter=epoch_iter % dataset_size
    for i, data in enumerate(dataloader, start=epoch_iter):
        iter_start_time=time.time()
        total_steps+=opt.batchSize
        epoch_iter+=opt.batchSize
        total_steps += opt.batchSize
        # seg2 and seg2_ori are different
        model(Variable(data['land1']), Variable(data['ori1']),
                  Variable(data['land2']), Variable(data['ori2']))
        losses, fake_images=model.module.optimize_parameters()

        loss_dict =dict(zip(model.module.loss_names, losses))

        # print out errors
        if total_steps % opt.print_freq==0:
            errors={k: v  for k , v in loss_dict.items()}
            t=(time.time() - iter_start_time ) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        # display output images
        save_fake=total_steps % opt.display_freq==0
        #######
        # if save_fake:
            # for j, data2 in enumerate(test_dataloader, start=0):
            #     if j>=opt.how_many:
            #         break
            #     generated=model.module.inference(Variable(data2['land2']), Variable(data2['ori1']))
            #     test_visuals = OrderedDict([
            #         ('test_land1', util.tensor2im(data2['land1'][0])),
            #         ('test_ori1', util.tensor2im(data2['ori1'][0])),
            #         ('test_land2', util.tensor2im(data2['land2'][0])),
            #         ('test_ori2', util.tensor2im(data2['ori2'][0])),
            #         ('test_A->B', util.tensor2im(generated.data[0]))
            #     ])

        if save_fake:
            if opt.use_seg:
                visuals= OrderedDict([
                    ('input_ori', util.tensor2im(data['ori1'][0])),
                    ('input_land', util.tensor2im(data['land1'][0])),
                    ('seg1_ori', util.tensor2im(data['seg1_ori'][0],normalize=False)),
                    ('output_land', util.tensor2im(data['land2'][0])),
                    ('seg2_ori', util.tensor2im(data['seg2_ori'][0], normalize=False)),
                    ('output_ori', util.tensor2im(data['ori2'][0])),
                    ('A->B', util.tensor2im(fake_images['fake_B'][0])),
                ])
            else:
                visuals = OrderedDict([
                    ('input_ori', util.tensor2im(data['ori1'][0])),
                    ('input_land', util.tensor2im(data['land1'][0])),
                    ('output_land', util.tensor2im(data['land2'][0])),
                    ('output_ori', util.tensor2im(data['ori2'][0])),
                    ('A->B', util.tensor2im(fake_images['fake_B'][0])),
                ])

            #####
            # visuals.update(test_visuals)

            visualizer.display_current_results(visuals, epoch, total_steps)


        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')


    # update learning rate
    if epoch > opt.niter:
        model.module.update_learning_rate()

