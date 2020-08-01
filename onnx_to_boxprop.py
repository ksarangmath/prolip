import numpy as np
import onnx
from onnx import numpy_helper
from onnx_translator import *
from boxprop_optimized import *


def boxprop():
  model = onnx.load('small_cifar.onnx')

  resources = {}

  for initial in model.graph.initializer:
    const = numpy_helper.to_array(initial)
    resources[initial.name] = const


  for node in model.graph.node:
    if node.op_type == 'Relu':
      print('b.relu()')

    if node.op_type == 'Conv':
      weight = None
      bias = None

      for inp in node.input:
        if inp.split('.')[-1] == 'weight':
          weight = resources[inp]
          # print(resources[inp])
        if inp.split('.')[-1] == 'bias':
          bias = resources[inp]
      c_out = weight.shape[0]

      kernel_size = [0,0]
      stride = 1
      for attr in node.attribute:
        if attr.name == 'kernel_shape':
          for i,ints in enumerate(attr.ints):
            kernel_size[i] = ints
        if attr.name == 'strides':
          stride = attr.ints[0]
      print('b.conv2d(weight,c_out,kernel_size,stride,bias)')

    if node.op_type == 'MaxPool':
      kernel_size = 1
      for attr in node.attribute:
        if attr.name == 'kernel_shape':
          kernel_size = attr.ints[0]
      print('b.maxpool2d(kernel_size)')

    if node.op_type == 'Gemm':
      weight = None
      bias = None

      for inp in node.input:
        if inp.split('.')[-1] == 'weight':
          weight = resources[inp]
          # print(resources[inp])
        if inp.split('.')[-1] == 'bias':
          bias = resources[inp]

      print('b.fc(weight,bias)')




  # print(resources)

    # if node.op_type == 'MaxPool':


    # if node.op_type == 'Gemm':

  # for init in model.graph.initializer:
  #   print(init)

  # for init in model.graph.node:
  #   print(init)





    

def main():

  # translator = ONNXTranslator(model)

  # operation_types, operation_resources = translator.translate()

  # print(operation_types, operation_resources)

  boxprop()

  # print(len(onnx_model.graph.node),len(onnx_model.graph.initializer))
	

if __name__ == "__main__":
    main()
