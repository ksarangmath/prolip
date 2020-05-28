import torch
import numpy as np
import csv
from numba import jit,prange

torch.manual_seed(0)

batch_size = 1
latent_size = 100

# with open('cifar10_results.csv', mode='w') as cifar10_file:
# 	fieldnames = ['center','size','time']
# 	writer = csv.DictWriter(cifar10_file, fieldnames=fieldnames)
# 	writer.writeheader()
# 	for _ in range(5):
# 		center = torch.randn(batch_size,latent_size,1,1)
# 		for size in [0.25,0.5,0.75]:
# 			upper_bound = center+size
# 			lower_bound = center-size
# 			writer.writerow({'center':center,'size':size,'time':0.342523453})

@jit(nopython=True,parallel=True)
def loop():
	one=[1 for i in range(0)]
	two=[1 for i in range(0)]
	three=[1 for i in range(0)]
	d = 0
	for i in prange(5):
		for j in prange(5):
			d+=1
			one.append(j)
			two.append(j+1)
			three.append(j+2)
	return one,two,three

print(loop())