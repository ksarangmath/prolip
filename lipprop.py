import numpy as np
import itertools
import torch
from numba import jit
import warnings
import cProfile
import math
from scipy import sparse
from numba import prange
# warnings.filterwarnings("ignore")


class Lip():


	#lip analysis only handles 1 batch currently
	def __init__(self, lowerShape, upperShape):
		self.lower = np.identity(np.product(lowerShape), dtype=np.float32)
		self.upper = np.identity(np.product(upperShape), dtype=np.float32)
		self.sparse = True

	def conv2d(self, weights, c_out, kernel_size, stride, h_out, w_out, inp_shape, padding=0):
		shape = self.upper.shape
		# lower = np.empty((np.product((c_out,h_out,w_out)), np.product(inp_shape)))
		inp_shape1=list(inp_shape)[1]
		inp_shape2=list(inp_shape)[2]
		inp_shape3=list(inp_shape)[3]
		w0 = len(weights)
		w1 = len(weights[0])
		i_index = (c_out*h_out*w_out)//w0
		j_index = (inp_shape[0]*inp_shape[1]*inp_shape[2]*inp_shape[3])//w1

		@jit(nopython=True, parallel=True)
		def conv():
			# weights = np.float32(weights)

			upper = np.empty((c_out*h_out*w_out, shape[0]),dtype=np.float32)

			for i in prange(w0):
				for j in prange(w1):
					for p in prange(h_out):
						for q in prange(w_out):
							temp = np.zeros((inp_shape2+padding*2,inp_shape3+padding*2),dtype=np.float32)
							temp[p*stride:p*stride+weights[i][j].shape[0],q*stride:q*stride+weights[i][j].shape[1]] = weights[i][j]
							temp = temp[padding:-padding,padding:-padding] if padding != 0 else temp
							upper[i*i_index + p*h_out + q,j*j_index:(j+1)*j_index] = temp.flatten()
				# lower[i*i_index + p*h_out + q,j*j_index:(j+1)*j_index] = (temp*(temp<0)).flatten()

			return upper
		
		# print('W start')
		upper = conv()
		# print('W done')


		oldUpper = self.upper
		oldLower = self.lower

		if self.sparse:
			# print('converting to sparse')
			oldUpper = sparse.csr_matrix(self.upper)
			oldLower = sparse.csr_matrix(self.lower)
			upper = sparse.csr_matrix(upper)
			# print('done converting')
			# print('start dot')
			newUpper = ((upper.multiply(upper>0))@oldUpper) + ((upper.multiply(upper<0))@oldLower)
			newLower = ((upper.multiply(upper<0))@oldUpper) + ((upper.multiply(upper>0))@oldLower)

			# print(newUpper.count_nonzero()/(newUpper.shape[0]*newUpper.shape[1]), 'sparsity value')

			if (newUpper.getnnz()/(newUpper.shape[0]*newUpper.shape[1])) > 0.075 or newUpper.shape[0]*newUpper.shape[1] < 10000000:
				# if newUpper.shape[0]*newUpper.shape[1]<10000000:
				# 	print(newUpper.shape[0]*newUpper.shape[1], 'small matrix')
				# else:
				# 	print('not sparse')
				self.sparse = False

			newUpper = newUpper.toarray()
			newLower = newLower.toarray()

		else:
			# print('start dot')
			newUpper = ((upper*(upper>0))@oldUpper) + ((upper*(upper<0))@oldLower)
			newLower = ((upper*(upper<0))@oldUpper) + ((upper*(upper>0))@oldLower)
			# print(np.count_nonzero(newUpper)/(newUpper.shape[0]*newUpper.shape[1]), 'sparsity value')

		self.upper = newUpper
		self.lower = newLower

		
		return newLower

	def batchNorm2d(self, var, weights, eps):

		var=np.float32(var)

		if weights is not None:
			step = self.upper.shape[0]//len(var)
			for i in range(len(var)):
				if weights[i] >= 0:

					self.upper[i:i*step,:] *= (weights[i]/np.sqrt(var[i]+eps))
					self.lower[i:i*step,:] *= (weights[i]/np.sqrt(var[i]+eps))
				else:
					l = (self.upper[i:i*step,:])*(weights[i]/np.sqrt(var[i]+eps))
					u = (self.lower[i:i*step,:])*(weights[i]/np.sqrt(var[i]+eps))
					self.lower[i:i*step,:] = l
					self.upper[i:i*step,:] = u
		else:
			step = self.upper.shape[0]//len(var)
			for i in range(len(var)):
				self.upper[i*step:step+(1*i)] *= 1/math.sqrt(var[i]+eps)
				self.lower[i*step:step*(1+i)] *= 1/math.sqrt(var[i]+eps)

		

		return self.lower

		

	def maxpool2d(self, kernel_size, in_shape, h_out, w_out):

		upper = np.empty((in_shape[1]*h_out*w_out,self.upper.shape[1]),dtype=np.float32)
		lower = np.empty((in_shape[1]*h_out*w_out,self.lower.shape[1]),dtype=np.float32)

		for c in range(in_shape[1]):
			for h in range(0, in_shape[2], kernel_size):
				for w in range(0, in_shape[3], kernel_size):
					inds = [(c*in_shape[2]*in_shape[3])+((h+i)*in_shape[3])+(w+j) for i,j in itertools.product(range(kernel_size),range(kernel_size))]
					upper[c*h_out*w_out + (h//kernel_size)*w_out + w//kernel_size] = np.max(self.upper[inds],0)
					lower[c*h_out*w_out + (h//kernel_size)*w_out + w//kernel_size] = np.min(self.lower[inds],0)

		self.upper = upper
		self.lower = lower



	def linear(self, weights):
		posWeights = (weights*(weights>=0))
		negWeights = (weights*(weights<0))
		upper = posWeights.dot(self.upper) + negWeights.dot(self.lower)
		lower = posWeights.dot(self.lower) + negWeights.dot(self.upper)
		self.upper = upper
		self.lower = lower



	def relu(self, active, uncertain):
		for i in range(len(self.upper)):
			if i in active:
				continue
			elif i in uncertain:
				self.upper[i] = self.upper[i]*(self.upper[i]>=0)
				self.lower[i] = self.lower[i]*(self.lower[i]<0)
			else:
				self.upper[i][:] = 0
				self.lower[i][:] = 0


	def operatorNorm(self):
		maxabs = np.where(-self.lower > self.upper, self.lower, self.upper)
		return (np.linalg.norm(maxabs, ord=2))

