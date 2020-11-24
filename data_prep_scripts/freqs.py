import csv
import subprocess
import math
import sys

def freqs_func(csv_file, save_file, nframes, log_file):
	csv_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
	
	freqs = {}
	bashcmd = ["soxi", "-D"]
	for i, line in enumerate(csv_reader):
		path = line[0]
		label = int(line[1])
		if label not in freqs:
			freqs[label] = 0
		#audio, rate = torchaudio.load('/storage' + path)
		process = subprocess.Popen(bashcmd + ['/storage' + path], stdout=subprocess.PIPE)
		output, error = process.communicate()
		output = float(output.decode('utf-8').strip())
		num_parts = math.ceil(output * 8000 / nframes) 
		freqs[label] += num_parts

		if i % 1000 == 0:
			print("Still processing file(Line {})".format(i), file=log_file, flush=True)
			print(freqs, file=log_file, flush=True)
			ff = open(save_file, 'w')
			for item in freqs:
				ff.write(str(item) + " " + str(freqs[item]) + "\n")
			ff.close()
	
	ff = open(save_file, 'w')
	for item in freqs:
		ff.write(str(item) + " " + str(freqs[item]) + "\n")
	ff.close()

	return freqs

if __name__ == '__main__':
	freqs = freqs_func('data/final_equal_shuffled_train.csv', './data/freqs_equal.txt', 80000, sys.stdout)

