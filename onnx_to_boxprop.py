import numpy as np
import onnx
from onnx import numpy_helper
from boxprop import *



def boxprop(box, model):

  resources = {}

  for initial in model.graph.initializer:
    const = numpy_helper.to_array(initial)
    resources[initial.name] = const

  ignoredNodes = {'Constant', 'Reshape', 'Concat', 'Unsqueeze', 'Shape', 'Gather'}

  for node in model.graph.node:

    if node.op_type == 'Relu':
      # print('box.relu()')
      box.relu()


    elif node.op_type == 'Tanh':
      # print('box.tanh()')
      box.tanh()


    elif node.op_type == 'Conv':
      weight = None
      bias = None
      for inp in node.input:
        if inp.split('.')[-1] == 'weight':
          weight = resources[inp]
        if inp.split('.')[-1] == 'bias':
          bias = resources[inp]

      c_out = weight.shape[0]
      kernel_size = 1
      stride = 1
      padding = 0

      for attr in node.attribute:
        if attr.name == 'kernel_shape':
          kernel_size = attr.ints[0]
        elif attr.name == 'strides':
          stride = attr.ints[0]
        elif attr.name == 'pads':
          padding = attr.ints[0]
      # print('box.conv2d','weight',c_out,kernel_size[0],stride, padding,'bias')
      box.conv2d(weight,c_out,kernel_size,stride,padding,bias)


    elif node.op_type == 'ConvTranspose':
      weight = None
      for inp in node.input:
        if inp.split('.')[-1] == 'weight':
          weight = resources[inp]

      c_out = weight.shape[1]
      kernel_size = [0,0]
      stride = 1
      padding = 0

      for attr in node.attribute:
        if attr.name == 'kernel_shape':
          for i,ints in enumerate(attr.ints):
            kernel_size[i] = ints
        elif attr.name == 'strides':
          stride = attr.ints[0]
        elif attr.name == 'pads':
          padding = attr.ints[0]
      # print('box.convTranspose2d','weight', c_out, kernel_size, stride, padding)
      box.convTranspose2d(weight, c_out, kernel_size, stride, padding)


    elif node.op_type == 'BatchNormalization':
      mean = None
      var = None
      weight = None
      bias = None
      for inp in node.input:
        if inp.split('.')[-1] == 'weight':
          weight = resources[inp]
        elif inp.split('.')[-1] == 'bias':
          bias = resources[inp]
        elif inp.split('.')[-1] == 'running_mean':
          mean = resources[inp]
        elif inp.split('.')[-1] == 'running_var':
          var = resources[inp]

      eps = 0
      for attr in node.attribute:
        if attr.name == 'epsilon':
          eps = attr.f
      # print('box.batchNorm2d','mean', 'var', eps, weight, bias)
      box.batchNorm2d(mean, var, eps, weight, bias)


    elif node.op_type == 'MaxPool':
      kernel_size = 1
      for attr in node.attribute:
        if attr.name == 'kernel_shape':
          kernel_size = attr.ints[0]
      # print('box.maxpool2d',kernel_size)
      box.maxpool2d(kernel_size)


    elif node.op_type == 'Gemm':
      weight = None
      bias = None

      for inp in node.input:
        if inp.split('.')[-1] == 'weight':
          weight = resources[inp]
        if inp.split('.')[-1] == 'bias':
          bias = resources[inp]
      # print('box.linear','weight','bias')
      box.linear(weight,bias)


    elif node.op_type not in ignoredNodes:
      raise ValueError('Cannot handle layer of type ' + node.op_type)


   

# def main():

  # for node in model.graph.node:
  #   print(node)

#   boxprop(None, 'cifar_g.onnx')	

# if __name__ == "__main__":
#     main()
