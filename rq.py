import sys
sys.path.insert(0, "./pretrained_classifiers")
sys.path.insert(0, "./")
sys.path.insert(0, "./mnist")
sys.path.insert(0, "./small_cifar")
sys.path.insert(0, "./large_cifar")
import os
import torch
import numpy as np
import io
import time
import itertools
import csv
import matplotlib as mpl 
import datetime
from matplotlib import rc
import matplotlib.pyplot as plt
import importlib
import heapq

def main():

	gen = importlib.import_module('small_cifar_g')
	clf = importlib.import_module('small_cifar_f')

	G = gen.loadGen()
	C = clf.loadClf()

	randomSeed=1
	torch.manual_seed(randomSeed)

	center = torch.randn(1,100,1,1)

	size = 0.2

	upper_bound = center+size
	lower_bound = center-size

	tottic = time.perf_counter()
	a_o = gen.gen(upper_bound,lower_bound,G)
	ans = C(G(center))[0]
	print(ans)
	jacInds = heapq.nlargest(2, range(len(ans)), key=ans.__getitem__)
	print(jacInds)
	lipc, jac = clf.clf(a_o,C)
	totaltime=time.perf_counter()-tottic

	print(lipc, np.linalg.norm(jac[jacInds[-1]]),jac.shape, 'LIPSCHITZ')



if __name__ == "__main__":
    main()