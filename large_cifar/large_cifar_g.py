import sys
sys.path.insert(0, "./pretrained_generators/weights_cifar")
from dcgan_cifar import Generator
from boxprop_optimized import *
import torch

def loadGen():
	#load generator model here
	G = Generator(ngpu=1).eval()
	G.load_state_dict(torch.load('pretrained_generators/weights_cifar/netG_epoch_199.pth',map_location=torch.device('cpu')))
	return G


def gen(upper_bound, lower_bound, G):

	a_o = Box_o(upper_bound,lower_bound,False)

	a_o.convTranspose2d(weight=G.state_dict()['main.0.weight'], c_out=512, kernel_size=4, stride=1, padding=0, output_padding=0)
	a_o.batchNorm2d(mean=G.state_dict()['main.1.running_mean'], var=G.state_dict()['main.1.running_var'], eps=1e-05, weight=G.state_dict()['main.1.weight'], bias=G.state_dict()['main.1.bias'])
	a_o.relu()
	
	a_o.convTranspose2d(weight=G.state_dict()['main.3.weight'], c_out=256, kernel_size=4, stride=2, padding=1, output_padding=0)
	a_o.batchNorm2d(mean=G.state_dict()['main.4.running_mean'], var=G.state_dict()['main.4.running_var'], eps=1e-05, weight=G.state_dict()['main.4.weight'], bias=G.state_dict()['main.4.bias'])
	a_o.relu()
	
	a_o.convTranspose2d(weight=G.state_dict()['main.6.weight'], c_out=128, kernel_size=4, stride=2, padding=1, output_padding=0)
	a_o.batchNorm2d(mean=G.state_dict()['main.7.running_mean'], var=G.state_dict()['main.7.running_var'], eps=1e-05, weight=G.state_dict()['main.7.weight'], bias=G.state_dict()['main.7.bias'])
	a_o.relu()
	
	a_o.convTranspose2d(weight=G.state_dict()['main.9.weight'], c_out=64, kernel_size=4, stride=2, padding=1, output_padding=0)
	a_o.batchNorm2d(mean=G.state_dict()['main.10.running_mean'], var=G.state_dict()['main.10.running_var'], eps=1e-05, weight=G.state_dict()['main.10.weight'], bias=G.state_dict()['main.10.bias'])
	a_o.relu()
	
	a_o.convTranspose2d(weight=G.state_dict()['main.12.weight'], c_out=3, kernel_size=1, stride=1, padding=0, output_padding=0)
	a_o.tanh()
	
	print("generator propagation done")
	return a_o
