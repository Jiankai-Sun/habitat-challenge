import numpy as np
import os, sys
from constants import *
from model_fpn import I2D
import argparse, time
from utils.net_utils import adjust_learning_rate
import torch
from torch.autograd import Variable
# from dataset.dataloader import DepthDataset
from dataset.nyuv2_dataset import NYUv2Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from collections import Counter
import matplotlib, cv2
matplotlib.use('Agg')
from model_fpn import I2D


if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You might want to run with --cuda")

# network initialization
print('Initializing model...')
i2d = I2D(fixed_feature_weights=False)
i2d = i2d.cuda()
print('Done!')

# hyperparams
lr = args.lr
bs = args.bs
lr_decay_step = args.lr_decay_step
lr_decay_gamma = args.lr_decay_gamma

rmse = RMSE()
depth_criterion = RMSE_log()
grad_criterion = GradLoss()
normal_criterion = NormalLoss()
eval_metric = RMSE_log()

load_name = os.path.join(args.output_dir,
  'i2d_1_{}.pth'.format(args.checkepoch))

print("loading checkpoint %s" % (load_name))
state = i2d.state_dict()
checkpoint = torch.load(load_name)
checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
state.update(checkpoint)
i2d.load_state_dict(state)
if 'pooling_mode' in checkpoint.keys():
    POOLING_MODE = checkpoint['pooling_mode']
print("loaded checkpoint %s" % (load_name))
del checkpoint
torch.cuda.empty_cache()

i2d.eval()

img = Variable(torch.FloatTensor(1), volatile=True)
z = Variable(torch.FloatTensor(1), volatile=True)

img = img.cuda()
z = z.cuda()

img.data.resize_(data[0].size()).copy_(data[0])
z.data.resize_(data[1].size()).copy_(data[1])

z_fake = i2d(img)


       

