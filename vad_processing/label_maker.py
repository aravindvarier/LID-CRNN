import csv
import torchaudio
import math

csv_file = 'data/final_equal_shuffled_train.csv'
nframes = 80000
f = open('labels.txt', 'w')
csv_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
for line in csv_reader:
	path = line[0]
	label = line[1]
	audio, sample_rate = torchaudio.load('/storage' + path)
	audio_len = audio.shape[1]
	num_parts = math.ceil(audio_len / nframes)
	for i in range(num_parts):
		f.write(str(label) + "\n")
		f.flush()


