import sys
sys.path.insert(0, "./pretrained_classifiers")
sys.path.insert(0, "./")
from boxprop_optimized import *
from utee import selector
import torch
from torch import nn

def loadClf():

	#Load your pytorch classifier model in eval form and return it in this function
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
	return C


def clf(a_o,C):

	#Do not modify this line
	b = Box_o(a_o.upper, a_o.lower, True)

	#For each layer in your model, call the corresponding function from boxprop_optimized
	#C is the classifier model, use its state_dict to get trained model parameters like shown
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

	print('classifier propagation done')
	return b.getLip()








