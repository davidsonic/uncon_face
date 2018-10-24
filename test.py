import os
from collections import OrderedDict
from data.data_loader import CreateDataLoader
from model.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from options.test_options import TestOptions
from torch.autograd import Variable
import pdb

opt = TestOptions().parse()
opt.nThreads=1
opt.batchSize=1
opt.serial_batches=False # no shuffle
opt.no_flip=True

data_loader=CreateDataLoader(opt)
dataset=data_loader.load_data()
print 'dataset_size: {}'.format(len(data_loader))
model=create_model(opt)
visualizer=Visualizer(opt)

web_dir=os.path.join(opt.results_dir, opt.name, '%s_%s'  %(opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

for i, data in enumerate(dataset):
    if i>=opt.how_many:
        break

    # inference
    generated = model.inference(Variable(data['land2']), Variable(data['ori1']))
    # pdb.set_trace()
    visuals  = OrderedDict([
        ('land1', util.tensor2im(data['land1'][0])),
        ('ori1', util.tensor2im(data['ori1'][0])),
        ('land2', util.tensor2im(data['land2'][0])),
        ('ori2', util.tensor2im(data['ori2'][0])),
        ('A->B', util.tensor2im(generated.data[0]))
    ])
    visualizer.save_inference_images(webpage, visuals)

webpage.save()