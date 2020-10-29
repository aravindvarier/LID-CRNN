import csv
import subprocess
import math

def freqs_func(csv_file, nframes):
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
			print("Still processing file(Line {})".format(i))
			print(freqs)
			ff = open('data/freqs_subset.txt', 'w')
			for item in freqs:
				ff.write(str(item) + " " + str(freqs[item]) + "\n")
			ff.close()
	return freqs

freqs = freqs_func('data/final_subset_shuffled_train.csv', 80000)

