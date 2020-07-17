import csv
import os
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image 

class audio_dataset(Dataset):
    
    def __init__(self, root, csv_file):
        langs = os.listdir(root)
        self.lang2id = {lang: i for i,lang in enumerate(langs)}
        self.csv_filename = csv_file
        # self.csv_filename = csv_file

        with open(self.csv_filename, 'r') as a2l:
            self.audio2label = a2l.readlines()

    def __len__(self):
        return(len(self.audio2label))

    def __getitem__(self, idx):
        file_name, label = self.audio2label[idx].split(',')
        label = torch.tensor(int(label))
        # audio_npy = torch.tensor(np.load(file_name)).unsqueeze(0)
        img_file_name = os.path.splitext(file_name)[0] + ".png"
        command = "sox -V0 {} -n remix 1 rate 8k spectrogram -y {} -X {} -m -r -o {}".format(file_name, 129, 100, img_file_name)
        os.system(command)
        img = Image.open(img_file_name)
        spectrogram = torch.tensor(img)
        command = "rf -rf {}".format(img_file_name)
        os.system(command)
        return spectrogram, label
        
