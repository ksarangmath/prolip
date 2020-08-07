import os
import torch
import numpy as np
import time
import csv
import matplotlib as mpl 
from matplotlib import rc
import matplotlib.pyplot as plt
import argparse
import onnx
from onnx import numpy_helper
from boxprop import Box
from onnx_to_boxprop import boxprop

def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.onnx']:
        raise argparse.ArgumentTypeError('only .onnx format supported')
    return fname

def gen(upper_bound, lower_bound, G):
	gen_box = Box(upper_bound,lower_bound, False)
	boxprop(gen_box, G)
	print("generator propagation done")
	return gen_box

def clf(gen_box, C):
	clf_box = Box(gen_box.upper, gen_box.lower, True)
	boxprop(clf_box, C)
	print('classifier propagation done')
	return clf_box


def main():
	parser = argparse.ArgumentParser(description='Arguments for PROLIP experiments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--genname', type=isnetworkfile, default=None, help='the generator network name, must be an onnx network with .onnx extension')
	parser.add_argument('--clfname', type=isnetworkfile, default=None, help='the classifier network name, must be an onnx network with .onnx extension')
	parser.add_argument('--boxsizes', type=int, default=[0.00001,0.001,0.1], nargs='+', help='list of box sizes')
	parser.add_argument('--numcenters', type=int, default=1, help='number of random centers')
	parser.add_argument('--randomseed', type=int, default=0, help='torch random seed for picking random box centers')
	parser.add_argument('--outfile', type=str, default='out', help='name for output files')
	args = parser.parse_args()

	assert args.genname, 'a generator network has to be provided for analysis.'
	assert args.clfname, 'a classifier network has to be provided for analysis.'

	rc('font', **{'serif': ['Computer Modern']})
	mpl.rcParams.update({'font.size': 14})

	G = onnx.load(args.genname)
	C = onnx.load(args.clfname)	    

	latent_size = numpy_helper.to_array(G.graph.initializer[0]).shape[0]
	boxSizes = args.boxsizes
	numCenters = args.numcenters
	randomSeed = args.randomseed
	filename = args.outfile

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
				a_o = gen(upper_bound,lower_bound,G)
				lipc = clf(a_o, C).getLip()
				totaltime=time.perf_counter()-tottic

				print('total time:', totaltime, 'lipc:', lipc)
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
    main()
	















