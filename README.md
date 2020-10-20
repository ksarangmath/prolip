The ```PROLIP``` algorithm analyzes the behavior of a neural network in a user-specified box-shaped input region and computes: 
(i) lower bounds on the probabilistic mass of such a region with respect to the generative model
(ii) upper bounds on the Lipschitz constant of the neural network in this region, with the help of the local Lipschitzness analysis.

Installation
------------
Clone the repository via git as follows:
```
git clone https://github.com/ksarangmath/prolip.git
cd prolip
```

Run the following command to install the dependencies:

```
pip3 install -r requirements.txt
```


Usage
-------------

```
python3 experiment.py --genname <path name to generator network> --clfname <path name to classifier network> --boxsizes <list of ints> --numcenters <int greater than 0> --randomseed <int> --outfile <name of output files>
```

* ```<genname>```: path name to ONNX generator model (must have .onnx extension)

* ```<clfname>```: path name to ONNX classifier model (must have .onnx extension)

* ```<boxsizes>```: list of integers specifying the different box sizes to run the ```PROLIP``` algorithm on (default is 0.0001 0.001 0.1)

* ```<numcenters>```: specifies the number of random centers of boxes to run the ```PROLIP``` algorithm on (default is 1)

* ```<randomseed>```: specifies the seed for generating random tensors corresponding to the centers of each random box (default is 0)

* ```<outfile>```: name of output .png and .csv files (default is out)


**Note that the networks provided must be ONNX files.**
To convert a PyTorch model to ONNX, see https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html.
To convert a TensorFlow model to ONNX, see https://github.com/onnx/tensorflow-onnx.

Examples
-------------

MNIST ```PROLIP``` analysis
```
python3 experiment.py --genname ./pretrained_generators/mnist/mnist_g.onnx --clfname ./pretrained_classifiers/mnist/mnist_f.onnx --numcenters 5 --outfile mnist
```

CIFAR ```PROLIP``` analyses
```
python3 experiment.py --genname ./pretrained_generators/cifar/cifar_g.onnx --clfname ./pretrained_classifiers/cifar/small_cifar_f.onnx --numcenters 5 --outfile small_cifar
```
```
python3 experiment.py --genname ./pretrained_generators/cifar/cifar_g.onnx --clfname ./pretrained_classifiers/cifar/large_cifar_f.onnx --numcenters 5 --outfile large_cifar
```




File Directory
-------------
* ```experiment.py``` is used to interact with our tool. It runs all the experiments, more details are in "instructions.pdf".

* ```boxprop.py``` contains the code for the ```PROLIP``` box analysis instructions.

* ```lipprop.py``` is a helper class that is called by "boxprop.py", and it contains code for the PROLIP Lipschitz analysis instructions.

* ```onnx_to_boxprop.py``` contains a helper function that propagates a Box object (defined in ```boxprop.py```) through a given ONNX model.

* ```pretrained_generators``` contains one CIFAR-10 generator and one MNIST generator (models from https://github.com/csinva/gan-vae-pretrained-pytorch).

* ```pretrained_classifiers``` contains two CIFAR-10 classifiers and one MNIST classifier (models from https://github.com/aaron-xichen/pytorch-playground and https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

* ```precomputed_results``` contains results from the three given examples above

* ```requirements.txt``` lists the required python dependencies

