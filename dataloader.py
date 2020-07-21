import csv
import os
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image 
from torchvision import transforms

class audio_dataset(Dataset):
	def __init__(self, root, csv_file):
		langs = os.listdir(root)
		self.lang2id = {lang: i for i,lang in enumerate(langs)}
		self.csv_filename = csv_file
		
		with open(self.csv_filename, 'r') as a2l:
			self.audio2label = a2l.readlines()

	def __len__(self):
		return(len(self.audio2label))

	def __getitem__(self, idx):
		file_name, label = self.audio2label[idx].split(',')
		label = torch.tensor(int(label))
		img = Image.open(file_name)
		spectrogram = transforms.ToTensor()(img)
		return spectrogram, label
        
