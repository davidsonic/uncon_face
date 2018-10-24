import os
from collections import OrderedDict
from data.data_loader import CreateDataLoader
from model.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from options.total_options import TotalOptions
from torch.autograd import Variable
import time

opt = TotalOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = False
opt.no_flip = True

# Load one image for test
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print 'dataset size: {}'.format(dataset_size)
# init model
model = create_model(opt)
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
#
for i, data in enumerate(dataset):
    if i >= 4:
        break

    # online adaptation
    iter_start_time=time.time()
    for t in range(200):
        model(Variable(data['land1']), Variable(data['ori1']),
              Variable(data['land2']), Variable(data['ori2']))
        losses, fake_images = model.module.optimize_parameters()
        loss_dict = dict(zip(model.module.loss_names, losses))


    elapse=time.time()-iter_start_time
    print 'The {}th image finetuned 200 iters, elapsed {}'.format(i,elapse)
    # display
    visuals = OrderedDict([
        ('land1', util.tensor2im(data['land1'][0])),
        ('ori1', util.tensor2im(data['ori1'][0])),
        ('land2', util.tensor2im(data['land2'][0])),
        ('ori2', util.tensor2im(data['ori2'][0])),
        ('A->B', util.tensor2im(fake_images['fake_B'][0]))
    ])

    visualizer.display_current_results(visuals, i, 5)
    visualizer.save_inference_images(webpage, visuals)

model.module.save('latest_finetune')
webpage.save()
