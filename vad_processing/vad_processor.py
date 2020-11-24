import csv
import numpy as np


with open('vad_results.txt') as f:
	vad_results = f.readlines()
	vad_results = [float(line.strip()) for line in vad_results]
with open('labels.txt') as f:
	labels = f.readlines()
	labels = [int(line.strip()) for line in labels]

vad_results = np.array(vad_results)

indices = np.where(vad_results <= 10)[0]
query_lines = []
for index in indices:
	query_lines.append(labels[index])

#print(len(wrong_indices))
print(query_lines.count(0), query_lines.count(1))
