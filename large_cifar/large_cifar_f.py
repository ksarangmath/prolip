import sys
sys.path.insert(0, "./pretrained_classifiers")
sys.path.insert(0, "./")
from boxprop_optimized import *
from utee import selector
import torch

def loadClf():
	model_raw, _, _ = selector.select('cifar10')
	model_raw.eval()
	C = model_raw
	return C


def clf(a_o,C):

	b = Box_o(a_o.upper, a_o.lower, True)
	b.conv2d(weight=C.state_dict()['features.0.weight'], c_out=128, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.0.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.1.running_mean'], var=C.state_dict()['features.1.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	b.conv2d(weight=C.state_dict()['features.3.weight'], c_out=128, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.3.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.4.running_mean'], var=C.state_dict()['features.4.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	b.maxpool2d(2)
	b.conv2d(weight=C.state_dict()['features.7.weight'], c_out=256, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.7.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.8.running_mean'], var=C.state_dict()['features.8.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	b.conv2d(weight=C.state_dict()['features.10.weight'], c_out=256, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.10.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.11.running_mean'], var=C.state_dict()['features.11.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	b.maxpool2d(2)
	b.conv2d(weight=C.state_dict()['features.14.weight'], c_out=512, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.14.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.15.running_mean'], var=C.state_dict()['features.15.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	b.conv2d(weight=C.state_dict()['features.17.weight'], c_out=512, kernel_size=3, stride=1, padding=1, bias=C.state_dict()['features.17.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.18.running_mean'], var=C.state_dict()['features.18.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	b.maxpool2d(2)
	b.conv2d(weight=C.state_dict()['features.21.weight'], c_out=1024, kernel_size=3, stride=1, padding=0, bias=C.state_dict()['features.21.bias'])
	b.batchNorm2d(mean=C.state_dict()['features.22.running_mean'], var=C.state_dict()['features.22.running_var'], eps=1e-05, weight=None, bias=None)
	b.relu()

	b.maxpool2d(2)
	b.linear(weight=C.state_dict()['classifier.0.weight'], bias=C.state_dict()['classifier.0.bias'])


	print('classifier propagation done')
	return b.getLip()








