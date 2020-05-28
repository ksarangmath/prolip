import sys
sys.path.insert(0, "./pretrained_classifiers")
sys.path.insert(0, "./")
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import io
import torch.utils.model_zoo as model_zoo
import torch.onnx
import cProfile
import time
import itertools
import torch
from torch.autograd import Variable
from utee import selector
from boxprop_optimized import *
import csv

num_gpu = 1 if torch.cuda.is_available() else 0

# load the models
from dcgan_cifar import Discriminator, Generator

G = Generator(ngpu=1).eval()

# load weights
G.load_state_dict(torch.load('weights_cifar/netG_epoch_199.pth',map_location=torch.device('cpu')))
if torch.cuda.is_available():
    G = G.cuda()

batch_size = 1
latent_size = 100

model_raw, ds_fetcher, is_imagenet = selector.select('cifar10')

model_raw.eval()

torch.manual_seed(0)

batch_size = 1
latent_size = 100

def prolip(upper_bound, lower_bound):
	print(0)
	tic = time.perf_counter()
	a_o = Box_o(upper_bound,lower_bound,False)
	a_o.convTranspose2d(weight=G.state_dict()['main.0.weight'], c_out=512, kernel_size=4, stride=1, padding=0, output_padding=0)
	a_o.batchNorm2d(mean=G.state_dict()['main.1.running_mean'], var=G.state_dict()['main.1.running_var'], eps=1e-05, weight=G.state_dict()['main.1.weight'], bias=G.state_dict()['main.1.bias'])
	a_o.relu()
	print(time.perf_counter()-tic)
	print(1)
	tic = time.perf_counter()
	a_o.convTranspose2d(weight=G.state_dict()['main.3.weight'], c_out=256, kernel_size=4, stride=2, padding=1, output_padding=0)
	a_o.batchNorm2d(mean=G.state_dict()['main.4.running_mean'], var=G.state_dict()['main.4.running_var'], eps=1e-05, weight=G.state_dict()['main.4.weight'], bias=G.state_dict()['main.4.bias'])
	a_o.relu()
	print(time.perf_counter()-tic)
	print(2)
	tic = time.perf_counter()
	a_o.convTranspose2d(weight=G.state_dict()['main.6.weight'], c_out=128, kernel_size=4, stride=2, padding=1, output_padding=0)
	a_o.batchNorm2d(mean=G.state_dict()['main.7.running_mean'], var=G.state_dict()['main.7.running_var'], eps=1e-05, weight=G.state_dict()['main.7.weight'], bias=G.state_dict()['main.7.bias'])
	a_o.relu()
	print(time.perf_counter()-tic)
	print(3)
	tic = time.perf_counter()
	a_o.convTranspose2d(weight=G.state_dict()['main.9.weight'], c_out=64, kernel_size=4, stride=2, padding=1, output_padding=0)
	a_o.batchNorm2d(mean=G.state_dict()['main.10.running_mean'], var=G.state_dict()['main.10.running_var'], eps=1e-05, weight=G.state_dict()['main.10.weight'], bias=G.state_dict()['main.10.bias'])
	a_o.relu()
	print(time.perf_counter()-tic)
	print(4)
	tic = time.perf_counter()
	a_o.convTranspose2d(weight=G.state_dict()['main.12.weight'], c_out=3, kernel_size=1, stride=1, padding=0, output_padding=0)
	a_o.tanh()
	print(time.perf_counter()-tic)
	print(5)
	print("done box prop")

	C = model_raw
	print(0)
	tic = time.perf_counter()
	b = Box_o(a_o.upper, a_o.lower, True)
	b.conv2d(weight=C.state_dict()['features.0.weight'], c_out=128, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.0.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.1.running_mean'], var=C.state_dict()['features.1.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	print(time.perf_counter()-tic)
	print(1)
	tic = time.perf_counter()

	b.conv2d(weight=C.state_dict()['features.3.weight'], c_out=128, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.3.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.4.running_mean'], var=C.state_dict()['features.4.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	print(time.perf_counter()-tic)
	print(2)
	tic = time.perf_counter()

	b.maxpool2d(2)
	b.conv2d(weight=C.state_dict()['features.7.weight'], c_out=256, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.7.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.8.running_mean'], var=C.state_dict()['features.8.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	print(time.perf_counter()-tic)
	print(3)
	tic = time.perf_counter()

	b.conv2d(weight=C.state_dict()['features.10.weight'], c_out=256, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.10.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.11.running_mean'], var=C.state_dict()['features.11.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	print(time.perf_counter()-tic)
	print(4)
	tic = time.perf_counter()

	b.maxpool2d(2)
	b.conv2d(weight=C.state_dict()['features.14.weight'], c_out=512, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.14.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.15.running_mean'], var=C.state_dict()['features.15.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	print(time.perf_counter()-tic)
	print(5)
	tic = time.perf_counter()

	b.conv2d(weight=C.state_dict()['features.17.weight'], c_out=512, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.17.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.18.running_mean'], var=C.state_dict()['features.18.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	print(time.perf_counter()-tic)
	print(6)
	tic = time.perf_counter()

	b.maxpool2d(2)
	b.conv2d(weight=C.state_dict()['features.21.weight'], c_out=1024, kernel_size=3, stride=1, padding=0, bias=C.state_dict()['features.21.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.22.running_mean'], var=C.state_dict()['features.22.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	print(time.perf_counter()-tic)
	print(7)
	tic = time.perf_counter()

	b.maxpool2d(2)
	b.linear(weight=C.state_dict()['classifier.0.weight'], bias=C.state_dict()['classifier.0.bias'])

	return b.getLip()
	print(time.perf_counter()-tic)
	print(8)


with open('largecifar_results.csv', mode='w') as cifar10_file:
	fieldnames = ['center','size','lip-constant','time']
	writer = csv.DictWriter(cifar10_file, fieldnames=fieldnames)
	writer.writeheader()
	for _ in range(5):
		center = torch.randn(batch_size,latent_size,1,1)
		for size in [0.00001,0.001,0.1]:
			upper_bound = center+size
			lower_bound = center-size

			tottic = time.perf_counter()
			lipc = prolip(upper_bound,lower_bound)
			totaltime=time.perf_counter()-tottic

			print(totaltime, 'total time', 'lipc:', lipc)
			print('ROUND DONE')
			writer.writerow({'center':center,'size':size,'lip-constant':lipc,'time':totaltime})












