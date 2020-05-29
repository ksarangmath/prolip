import sys
sys.path.insert(0, "./pretrained_classifiers")
sys.path.insert(0, "./")
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



def main(argv):
	gen = importlib.import_module(argv[0].split(".")[0])
	latent_size = int(argv[1])
	clf = importlib.import_module(argv[2].split(".")[0])
	numSizes = int(argv[3])
	boxSizes = [float(i) for i in argv[4:-3]]
	numCenters = int(argv[-3])
	randomSeed = int(argv[-2])
	filename = argv[-1].split(".")[0]

	rc('font', **{'serif': ['Computer Modern']})
	# rc('text', usetex=True)
	mpl.rcParams.update({'font.size': 14})

	G = gen.loadGen()
	C = clf.loadClf()

	torch.manual_seed(randomSeed)

	with open(filename +'.csv', mode='w') as file:
		fieldnames = ['center','size','lip-constant','time']
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		for _ in range(numCenters):
			center = torch.randn(1,latent_size,1,1)
			for size in boxSizes:
				upper_bound = center+size
				lower_bound = center-size

				tottic = time.perf_counter()
				a_o = gen.gen(upper_bound,lower_bound,G)
				lipc = clf.clf(a_o,C)
				totaltime=time.perf_counter()-tottic

				print(totaltime, 'total time', 'lipc:', lipc)
				print('ROUND DONE')
				writer.writerow({'center':center,'size':size,'lip-constant':lipc,'time':totaltime})



	sizeTime = {}

	with open(filename + '.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				print(f'Column names are {", ".join(row)}')
				line_count += 1
			else:

				if float(row[1]) not in sizeTime:
					sizeTime[float(row[1])] = [float(row[3])]
				else:
					sizeTime[float(row[1])].append(float(row[3]))

				line_count += 1
		print(f'Processed {line_count} lines.')

	x = [x+1 for x in range(numCenters)]



	N = len(x)
	ind = np.arange(N)  
	width = 0.24    # adjust this width if bars are too wide or narrow

	fig = plt.figure()
	ax = fig.add_subplot(111)

	handles = []

	keys = list(sizeTime.keys())
	keys.sort()
	for i,s in enumerate(keys):
		rects = ax.bar(ind+width*i, sizeTime[s], width,edgecolor='black')
		handles.append(rects)

	ax.set_xticks(ind+width)
	ax.set_xticklabels( x )
	ax.legend( [h[0] for h in handles], boxSizes ,title='Box Sizes', loc='lower right')

	plt.title('PROLIP Runtime on ' + filename)
	plt.ylabel('Runtime (seconds)')
	plt.xlabel('Random Centers')

	plt.savefig(filename+'.png', bbox_inches='tight')
    

if __name__ == "__main__":
    main(sys.argv[1:])
	















