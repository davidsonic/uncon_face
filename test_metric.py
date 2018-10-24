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
import time

opt = TestOptions().parse()
opt.nThreads=1
opt.batchSize=1
opt.serial_batches=True # no shuffle
opt.no_flip=True

data_loader=CreateDataLoader(opt)
dataset=data_loader.load_data()
print 'dataset_size: {}'.format(len(data_loader))
model=create_model(opt)
visualizer=Visualizer(opt)


begin=time.time()
for i, data in enumerate(dataset):
    if i>=opt.how_many:
        break

    # inference
    generated = model.inference(Variable(data['land2']), Variable(data['ori1']))
    # pdb.set_trace()
    save_path=os.path.join('results2', 'portraitGAN_nature2style', str(i)+'.png')
    print save_path
    util.save_image(util.tensor2im(generated.data[0]), save_path)

end=time.time()
print 'time used: {}'.format(end-begin)
print 'average time used: {}'.format((end-begin)/368)