import numpy as np
import torch.nn.functional as F
import torch
import itertools
from skimage.util.shape import view_as_windows
from lipprop import *



class Box():

	#upper and lower must be in NCHW layout where N, C, and W are 1
	def __init__(self, upper, lower, lip):
		self.upper = upper
		self.lower = lower
		if lip:
			self.lip = Lip(upper.shape, lower.shape)
		else:
			self.lip = 0
		

	def batchNorm2d(self, mean, var, eps, weight=None, bias=None):

		var = torch.from_numpy(var) if type(var) == np.ndarray else var
		mean = torch.from_numpy(mean) if type(mean) == np.ndarray else mean
		weight = torch.from_numpy(weight) if type(weight) == np.ndarray else weight
		bias = torch.from_numpy(bias) if type(bias) == np.ndarray else bias

		if weight is None:
			posWeight = None
			negWeight = None
			upper = F.batch_norm(input=self.upper, running_mean=mean, running_var=var, weight=posWeight, bias=bias, training=False, eps=eps)
			lower = F.batch_norm(input=self.lower, running_mean=mean, running_var=var, weight=negWeight, bias=bias, training=False, eps=eps)
		else:
			posWeight = weight*(weight>=0)
			negWeight = weight*(weight<0)
			upper = F.batch_norm(input=self.upper, running_mean=mean, running_var=var, weight=posWeight, bias=bias, training=False, eps=eps) + \
		 		F.batch_norm(input=self.lower, running_mean=mean, running_var=var, weight=negWeight, bias=None, training=False, eps=eps)
			lower = F.batch_norm(input=self.upper, running_mean=mean, running_var=var, weight=negWeight, bias=bias, training=False, eps=eps) + \
			 	F.batch_norm(input=self.lower, running_mean=mean, running_var=var, weight=posWeight, bias=None, training=False, eps=eps)

		self.upper = upper
		self.lower = lower

		# print(False in (self.upper==self.lower),'after bn')
		
		if self.lip != 0:
			return self.lip.batchNorm2d(var.numpy(), weight.numpy() if weight is not None else None, eps)




	def relu(self):
		# print(False in (self.upper==self.lower),'before relu')

		uncertain = set([])
		active = set([])


		upper = self.upper.reshape(np.product(self.upper.shape))
		lower = self.lower.reshape(np.product(self.lower.shape))
		
		# print(lower)
		

		for i in range(len(upper)):
			# print(upper[i],lower[i])
			if upper[i] < 0:
				upper[i] = 0
				lower[i] = 0

			elif lower[i] < 0:
				lower[i] = 0
				uncertain.add(i)
				
			else:
				active.add(i)

		self.upper = upper.reshape(self.upper.shape)
		self.lower = lower.reshape(self.lower.shape)

		if self.lip != 0:
			self.lip.relu(active,uncertain)

		return uncertain

		# print(False in (self.upper==self.lower),'after relu')
		# print(upper)
		# print(lower)


	def conv2d(self, weight, c_out, kernel_size, stride, padding=0, bias=None):

		weight = torch.from_numpy(weight) if type(weight) == np.ndarray else weight
		bias = torch.from_numpy(bias) if type(bias) == np.ndarray else bias

		inp_shape = self.upper.shape
		# print(inp_shape, 'inp shape')

		h_out = (self.upper.shape[2] + 2*padding - (kernel_size - 1) - 1)//stride + 1
		w_out = (self.upper.shape[3] + 2*padding - (kernel_size - 1) - 1)//stride + 1

		# print(h_out, w_out)

		posWeight = weight*(weight>=0)
		negWeight = weight*(weight<0)

		upper = F.conv2d(input=self.upper, weight=posWeight, bias=bias, stride=stride, padding=padding) + \
		 	F.conv2d(input=self.lower, weight=negWeight, bias=None, stride=stride, padding=padding)

		lower = F.conv2d(input=self.lower, weight=posWeight, bias=bias, stride=stride, padding=padding) + \
		 	F.conv2d(input=self.upper, weight=negWeight, bias=None, stride=stride, padding=padding)

		self.upper = upper
		self.lower = lower
		# print(self.upper.shape, 'out shape')


		if self.lip != 0:
			return self.lip.conv2d(weight.numpy(), c_out, kernel_size, stride, h_out, w_out, inp_shape, padding)

	#assumes no dilation, square kernel, square stride
	def convTranspose2d(self, weight, c_out, kernel_size, stride, padding=0, output_padding=0, bias=None):

		weight = torch.from_numpy(weight) if type(weight) == np.ndarray else weight

		# print(False in (self.upper==self.lower),'before conv')

		posWeight = weight*(weight>=0)
		negWeight = weight*(weight<0)

		upper = F.conv_transpose2d(input=self.upper, weight=posWeight, bias=bias, stride=stride, padding=padding) + \
		 	F.conv_transpose2d(input=self.lower, weight=negWeight, bias=None, stride=stride, padding=padding)

		lower = F.conv_transpose2d(input=self.lower, weight=posWeight, bias=bias, stride=stride, padding=padding) + \
		 	F.conv_transpose2d(input=self.upper, weight=negWeight, bias=None, stride=stride, padding=padding)


		self.upper = upper
		self.lower = lower
		# print(False in (self.upper==self.lower),'after conv')


	def maxpool2d(self, kernel_size, padding=0):

		h_out = (self.upper.shape[2] + 2*padding - (kernel_size - 1) - 1)//kernel_size + 1
		w_out = (self.upper.shape[2] + 2*padding - (kernel_size - 1) - 1)//kernel_size + 1

		inp_shape = self.upper.shape

		upper = F.max_pool2d(input=self.upper, kernel_size=kernel_size, padding=padding)
		lower = F.max_pool2d(input=self.lower, kernel_size=kernel_size, padding=padding)

		self.upper = upper
		self.lower = lower

		if self.lip != 0:
			self.lip.maxpool2d(kernel_size, inp_shape, h_out, w_out)

	#need to adjust input dimensions to NCHW format
	def linear(self, weight, bias):

		weight = torch.from_numpy(weight) if type(weight) == np.ndarray else weight
		bias = torch.from_numpy(bias) if type(bias) == np.ndarray else bias

		self.upper = self.upper.flatten()
		self.lower = self.lower.flatten()

		posWeight = (weight*(weight>=0))
		negWeight = (weight*(weight<0))

		upper = posWeight.matmul(self.upper) + negWeight.matmul(self.lower) + bias
		lower = negWeight.matmul(self.upper) + posWeight.matmul(self.lower) + bias

		self.upper = upper
		self.lower = lower

		if self.lip != 0:
			self.lip.linear(weight.numpy())


	def tanh(self):
		# print(False in (self.upper==self.lower))
		a = self.upper
		self.upper = np.tanh(self.upper)
		self.lower = np.tanh(self.lower)
		# print(False in (self.upper==self.lower))

	def getLip(self):
		return self.lip.operatorNorm()

	def getJac(self):
		return self.lip.jacobian()








