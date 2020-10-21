import csv
import numpy as np
import torch
import pandas as pd
import os
from numpy.random import shuffle
from PIL import Image
from torchvision import transforms

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
		
	
if __name__ == '__main__':
	langs = os.listdir('data/audio_data')
	lang2id = {lang: i for i,lang in enumerate(langs)}
	print(lang2id)
	for i, data in enumerate(my_dataloader('data/audio2label_train.csv', 1000)):
		audio, label = data
		print(i, audio.shape, label.shape)
