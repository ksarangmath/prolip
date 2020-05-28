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
import pylab
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
model.load_state_dict(torch.load('./smallcifar_net.pth'))
model.eval()
C = model


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

	b = Box_o(a_o.upper,a_o.lower,True)
	b.conv2d(weight=C.state_dict()['conv1.weight'],c_out=6,kernel_size=5,stride=1,bias=C.state_dict()['conv1.bias'])
	b.relu()
	b.maxpool2d(2)
	b.conv2d(weight=C.state_dict()['conv2.weight'],c_out=16,kernel_size=5,stride=1,bias=C.state_dict()['conv2.bias'])
	b.relu()
	b.maxpool2d(2)
	b.linear(weight=C.state_dict()['fc1.weight'],bias=C.state_dict()['fc1.bias'])
	b.relu()
	b.linear(weight=C.state_dict()['fc2.weight'],bias=C.state_dict()['fc2.bias'])
	b.relu()
	b.linear(weight=C.state_dict()['fc3.weight'],bias=C.state_dict()['fc3.bias'])
	return b.getLip()


# for _ in range(5):
# 	center = torch.randn(batch_size,latent_size,1,1)
# 	for size in [0.25,0.5,0.75]:
# 		upper_bound = center+size
# 		lower_bound = center-size
# 		tottic = time.perf_counter()
# 		print(time.perf_counter()-tottic, 'total time')
# 		print('ROUND DONE')




first = True
with open('smallcifar_results.csv', mode='w') as cifar10_file:
	fieldnames = ['center','size','lip-constant','time']
	writer = csv.DictWriter(cifar10_file, fieldnames=fieldnames)
	writer.writeheader()
	for _ in range(5):
		center = torch.randn(batch_size,latent_size,1,1)
		for size in [0.00001,0.001,0.1]:
			upper_bound = center+size
			lower_bound = center-size
			if first:
				lipc = prolip(upper_bound,lower_bound)
				first = False
			tottic = time.perf_counter()
			lipc = prolip(upper_bound,lower_bound)
			totaltime=time.perf_counter()-tottic

			print(totaltime, 'total time', 'lipc:', lipc)
			print('ROUND DONE')
			writer.writerow({'center':center,'size':size,'lip-constant':lipc,'time':totaltime})












