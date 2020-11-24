import csv
import numpy as np
import torch
import pandas as pd
import os
from numpy.random import shuffle
from PIL import Image
from torchvision import transforms
import torchaudio
from torchaudio.transforms import Spectrogram as spec, Vad, MelSpectrogram as melspec, SlidingWindowCmn as cmn 
import math
import subprocess
import random

np.random.seed(42)

class my_dataloader():
	def __init__(self, csv_file, batch_size):
		self.csv_file = csv_file
		self.batch_size = batch_size
		self.len = -1
		self.df = pd.read_csv(self.csv_file, header=None)	
		self.targets = self.df.iloc[:, 1].to_numpy()
	
		labels = np.unique(self.targets)
		self.num_classes = len(labels)
		
		self.class_indices = [list(np.where(self.targets == i)[0]) for i in labels]
		for item in self.class_indices:
			shuffle(item)
		self.class_freqs = [len(np.where(self.targets == i)[0]) for i in labels]	
		#print("Class freqs: ", class_freqs)
		
		self.max_freq = max(self.class_freqs)

		for i in range(self.num_classes):
			if self.class_freqs[i] < self.max_freq:
				n = int(self.max_freq / self.class_freqs[i])
				rem = self.max_freq % self.class_freqs[i]
				self.class_indices[i] = self.class_indices[i]*n + self.class_indices[i][:rem]


		self.samples_per_class = int(self.batch_size / self.num_classes)
	def __len__(self):
		return int(self.max_freq / self.samples_per_class)
	def __iter__(self):
		i = 0
		itr = 0
		
		while i < self.max_freq:
			all_indices = []
			all_spectrograms = []
			for item in self.class_indices:
				all_indices.extend(item[i:i+self.samples_per_class])
			file_names = self.df.iloc[all_indices, 0]
			true_labels = self.targets[all_indices]
			for file_name in file_names:
				img = Image.open(file_name)
				spectrogram = transforms.ToTensor()(img)
				all_spectrograms.append(spectrogram)
			data = torch.stack(all_spectrograms)
			i += self.samples_per_class
			itr += 1
			yield data, torch.tensor(true_labels)

class dataloader_v2:
	def __init__(self, csv_file, batch_size, num_seconds):
		self.csv_file = csv_file
		self.bs = batch_size
		self.nframes = int(8000 * num_seconds)
		self.total_samples = 0

	def __iter__(self):
		self.total_samples = 0
		def repeater(part):
			audio_len = part.shape[0]
			num_repeats = math.ceil(self.nframes / audio_len)
			part = part.tolist() * num_repeats
			return torch.tensor(part[:self.nframes])
		
		
		csv_reader = csv.reader(open(self.csv_file, 'r'), delimiter=',')
		all_audios = []	
		all_labels = []
		for line in csv_reader:
			path = line[0]
			label = line[1]
			audio, sample_rate = torchaudio.load('/storage' + path)
			audio_len = audio.shape[1]
			num_parts = math.ceil(audio_len / self.nframes)
			for i in range(num_parts):
				part = audio[:, self.nframes*i:self.nframes*(i+1)]
				part = part.squeeze(0)
				if part.shape[0] < self.nframes:
					part = repeater(part)
				all_audios.append(part)
				all_labels.append(int(label))
				if len(all_audios) == self.bs:
					self.total_samples += self.bs
					yield torch.stack(all_audios).unsqueeze(1), torch.tensor(all_labels)
					all_audios, all_labels = [], []
		
		if len(all_audios):
			self.total_samples += len(all_audios)
			yield torch.stack(all_audios).unsqueeze(1), torch.tensor(all_labels)

class dataloader_v3:
	def __init__(self, csv_file, batch_size, num_seconds):
		self.csv_file = csv_file
		self.batch_size = batch_size
		self.nframes = int(8000 * num_seconds)

	def __len__(self):
		with open(self.csv_file) as f:
			return len(f.readlines())

	def __iter__(self):
		def repeater(part):
			audio_len = part.shape[0]
			num_repeats = math.ceil(self.nframes / audio_len)
			part = part.tolist() * num_repeats
			return torch.tensor(part[:self.nframes])


		csv_reader = csv.reader(open(self.csv_file, 'r'), delimiter=',')
		all_audios = []
		all_labels = []
		for line in csv_reader:
			path = line[0]
			label = line[1]
			audio, sample_rate = torchaudio.load('/storage' + path)
			audio_len = audio.shape[1]
			audio = audio.squeeze(0)
			if audio_len < self.nframes:
				audio = repeater(audio)
			else:
				start = random.randint(0, audio_len - self.nframes)
				end = start + self.nframes
				audio = audio[start:end]
			all_audios.append(audio)
			all_labels.append(int(label))
			if len(all_audios) == self.batch_size:
				yield torch.stack(all_audios).unsqueeze(1), torch.tensor(all_labels)
				all_audios, all_labels = [], []
		if len(all_audios):
			yield torch.stack(all_audios).unsqueeze(1), torch.tensor(all_labels)


class val_dataloader:
	def __init__(self, csv_file):
		self.csv_file = csv_file
		self.transform = spec(256, 200)
	
	def __iter__():
		csv_reader = csv.reader(open(self.csv_file, 'r'), delimiter=',')
		for line in csv_reader:
			path = line[0]
			label = line[1]
			audio, _ = torchaudio.load('/storage' + path)
			spectrogram = self.transform(audio)
			yield spectrogram.unsqueeze(0).unsqueeze(0), torch.tensor(int(label))
			
		
	
if __name__ == '__main__':
	langs = os.listdir('data/audio_data')
	lang2id = {lang: i for i,lang in enumerate(langs)}
	print(lang2id)
	for i, data in enumerate(my_dataloader('data/audio2label_train.csv', 1000)):
		audio, label = data
		print(i, audio.shape, label.shape)
