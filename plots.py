import csv
import numpy as np
import matplotlib as mpl 
import datetime
import pandas as pd
from matplotlib import rc
rc('font', **{'serif': ['Computer Modern']})
rc('text', usetex=True)
import matplotlib.pyplot as plt


mpl.rcParams.update({'font.size': 14})

def subcategorybar(X, vals, width=0.8):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], 
                width=width/float(n), align="edge")   
    plt.xticks(_X, X)


size1time=[]
size2time=[]
size3time=[]
sizes = []
sizeSet = set([])
firstSize = -1
randomCenters = 0

with open('cifar10_results_final.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			print(f'Column names are {", ".join(row)}')
			line_count += 1
		else:
			if firstSize == -1:
				firstSize = float(row[1])
			if firstSize == float(row[1]):
				randomCenters += 1
			if row[1] == '1e-05':
				size1time.append(float(row[3]))
			elif row[1] == '0.001':
				size2time.append(float(row[3]))
			else:
				size3time.append(float(row[3]))

			if float(row[1]) not in sizeSet:
				sizeSet.add(float(row[1]))
				sizes.append(float(row[1]))
			# print(f'\t size:{row[1]} lipc: {row[2]} time: {row[3]}')
			line_count += 1
	print(f'Processed {line_count} lines.')

x = [x+1 for x in range(randomCenters)]



N = len(x)
ind = np.arange(N)  # the x locations for the groups
width = 0.24       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

handles = []

rects1 = ax.bar(ind, size1time, width,edgecolor='black')
handles.append(rects1)
rects2 = ax.bar(ind+width, size2time, width,edgecolor='black')
handles.append(rects2)
rects3 = ax.bar(ind+width*2, size3time, width,edgecolor='black')
handles.append(rects3)




ax.set_xticks(ind+width)
ax.set_xticklabels( x )
ax.legend( (rects1[0], rects2[0], rects3[0]), sizes ,title='Box Sizes')
plt.ylim(530,570)

plt.title('PROLIP Runtime on CIFAR-10 Program')
plt.ylabel('Runtime (seconds)')
plt.xlabel('Random Centers')

plt.savefig('cifar_results.png', bbox_inches='tight')


size1time=[]
size2time=[]
size3time=[]
sizes = []
sizeSet = set([])
firstSize = -1
randomCenters = 0

with open('mnist_results_final.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			print(f'Column names are {", ".join(row)}')
			line_count += 1
		else:
			if firstSize == -1:
				firstSize = float(row[1])
			if firstSize == float(row[1]):
				randomCenters += 1
			if row[1] == '1e-05':
				size1time.append(float(row[3]))
			elif row[1] == '0.001':
				size2time.append(float(row[3]))
			else:
				size3time.append(float(row[3]))

			if float(row[1]) not in sizeSet:
				sizeSet.add(float(row[1]))
				sizes.append(float(row[1]))
			# print(f'\t size:{row[1]} lipc: {row[2]} time: {row[3]}')
			line_count += 1
	print(f'Processed {line_count} lines.')

x = [x+1 for x in range(randomCenters)]



N = len(x)
ind = np.arange(N)  # the x locations for the groups
width = 0.24       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

handles = []

rects1 = ax.bar(ind, size1time, width,edgecolor='black')
handles.append(rects1)
rects2 = ax.bar(ind+width, size2time, width,edgecolor='black')
handles.append(rects2)
rects3 = ax.bar(ind+width*2, size3time, width,edgecolor='black')
handles.append(rects3)




ax.set_xticks(ind+width)
ax.set_xticklabels( x )
ax.legend( (rects1[0], rects2[0], rects3[0]), sizes ,title='Box Sizes')
plt.ylim(2,4.5)

plt.title('PROLIP Runtime on MNIST Program')
plt.ylabel('Runtime (seconds)')
plt.xlabel('Random Centers')

plt.savefig('mnist_results.png',bbox_inches='tight')




