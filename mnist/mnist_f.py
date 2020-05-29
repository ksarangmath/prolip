import sys
sys.path.insert(0, "./pretrained_classifiers")
sys.path.insert(0, "./")
from boxprop_optimized import *
from utee import selector
import torch

def loadClf():

	#Load your pytorch classifier model in eval form and return it in this function
	model_raw, _, _ = selector.select('mnist')
	model_raw.eval()
	C = model_raw
	return C


def clf(a_o,C):

	#Do not modify this line
	b = Box_o(a_o.upper, a_o.lower, True)

	#For each layer in your model, call the corresponding function from boxprop_optimized
	#C is the classifier model, use its state_dict to get trained model parameters like shown
	b.linear(C.state_dict()['model.fc1.weight'], C.state_dict()['model.fc1.bias'])
	b.relu()
	b.linear(C.state_dict()['model.fc2.weight'], C.state_dict()['model.fc2.bias'])
	b.relu()
	b.linear(C.state_dict()['model.out.weight'], C.state_dict()['model.out.bias'])

	print('classifier propagation done')
	return b.getLip()








