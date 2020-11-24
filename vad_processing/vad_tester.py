import csv
import torch
import torchaudio
from torchaudio.transforms import Spectrogram as spec, Vad
from torchaudio.transforms import MelSpectrogram as melspec
import math
import subprocess




csv_file = 'data/final_equal_shuffled_train.csv'
transform = spec(n_fft=256, win_length=200)
#self.transform = melspec(sample_rate=8000, n_fft=512, win_length=200, n_mels=129)
vad = Vad(sample_rate=8000)
nframes = 80000
f = open('vad_results.txt', 'w')

def repeater(part):
	audio_len = part.shape[0]
	num_repeats = math.ceil(nframes / audio_len)
	part = part.tolist() * num_repeats
	return torch.tensor(part[:nframes])


csv_reader = csv.reader(open(csv_file, 'r'), delimiter=',')
for line in csv_reader:
	path = line[0]
	label = line[1]
	audio, sample_rate = torchaudio.load('/storage' + path)
	audio_len = audio.shape[1]
	num_parts = math.ceil(audio_len / nframes)
	for i in range(num_parts):
		part = audio[:, nframes*i:nframes*(i+1)]
		part = part.squeeze(0)
		if part.shape[0] < nframes:
			part = repeater(part)
		spectrogram = transform(part)
		vad_decision = vad(spectrogram)
		f.write(str(vad_decision.sum().item()) + "\n")
		f.flush()


