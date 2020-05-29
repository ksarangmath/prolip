import sys
sys.path.insert(0, "./")
sys.path.insert(0, "./pretrained_generators/weights_mnist")
from boxprop_optimized import *
import torch
from dcgan_mnist import Generator


def loadGen():

	#Load your pytorch generator model in eval form and return it in this function
	G = Generator(ngpu=1).eval()
	G.load_state_dict(torch.load('pretrained_generators/weights_mnist/netG_epoch_99.pth', map_location=torch.device('cpu')))
	return G

def gen(upper_bound, lower_bound, G):

	#Do not modify this line
	a_o = Box_o(upper_bound,lower_bound, False)


	#For each layer in your model, call the corresponding function from boxprop_optimized
	#G is the generator model, use its state_dict to get trained model parameters like shown
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
	
	a_o.convTranspose2d(weight=G.state_dict()['main.12.weight'], c_out=1, kernel_size=1, stride=1, padding=2, output_padding=0)
	a_o.tanh()
	
	print("generator propagation done")
	return a_o