import torch.nn as nn
import torch
import numpy as np
#from torchvision import models
from my_inception import Inception3_small, Inception3_medium, Inception3_large
import math
from torchaudio.transforms import SlidingWindowCmn as Cmn, Spectrogram as Spec, MelSpectrogram as MelSpec


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VGG(nn.Module):
	def __init__(self):
		super(VGG, self).__init__()
		#min size - 102
		self.cnn = nn.Sequential(nn.Conv2d(1,16, (7,7)),
					nn.ReLU(),
					nn.BatchNorm2d(16),
					nn.MaxPool2d((2,2), 2),
					nn.Dropout(p=0.1),
                               	
					nn.Conv2d(16, 32, (5,5)),
					nn.ReLU(),
					nn.BatchNorm2d(32),
					nn.MaxPool2d((2,2), 2),
					nn.Dropout(p=0.1),

					nn.Conv2d(32,64,(3,3)),
					nn.ReLU(),
					nn.BatchNorm2d(64),
					nn.MaxPool2d((2,2), 2),
					nn.Dropout(p=0.1),
				
					nn.Conv2d(64,128,(3,3)),
					nn.ReLU(),
					nn.BatchNorm2d(128),
					nn.MaxPool2d((2,2), 2),
					nn.Dropout(p=0.1),
                                
					nn.Conv2d(128,256, (3,3)),
					nn.ReLU(),
					nn.BatchNorm2d(256),
					nn.MaxPool2d((2,2), 2),
					nn.Dropout(p=0.1),
					)
	def forward(self, x):
		return self.cnn(x)

class CRNN(nn.Module):
	def __init__(self, hidden_size, cnn_type, recurrent_type, lstm_layers, nheads, nlayers, input_shape, only_cnn=False):
		super(CRNN, self).__init__()
		self.hidden_size = hidden_size
		self.only_cnn = only_cnn
		self.cnn_type = cnn_type
		self.recurrent_type = recurrent_type #seeing as how transformers aren't recurrent, this variable has a very bad name, i know. but i honestly have no clue what to call it. for future readers of this code, please fix it!
		self.lstm_layers = lstm_layers
		self.cnn_out_d = -1
		self.cnn_out_n = -1
		self.cnn_ext_out_d = -1
		self.cnn_ext_out_n = -1
		
		if self.cnn_type == 'vgg':
			self.cnn = VGG()
		elif self.cnn_type == 'inceptionv3_l':
			#min size - 124
			self.cnn = Inception3_large()
		elif self.cnn_type == 'inceptionv3_m':
			self.cnn = Inception3_medium()
		else:
			self.cnn = Inception3_small()
		
		self.melspec = MelSpec(sample_rate=8000, n_fft=512, win_length=200, n_mels=129)
		self.cmn = Cmn()

		sample_audio = torch.randn(input_shape)
		sample_audio = self.melspec(sample_audio)
		if self.cnn_type == 'inceptionv3_l':
			with torch.no_grad():
				sample_output_ext, sample_output = self.cnn(sample_audio)
			self.cnn_ext_out_d = sample_output_ext.shape[1]
			self.cnn_ext_out_n = sample_output_ext.shape[-1]
		
		else:
			with torch.no_grad():
				sample_output = self.cnn(sample_audio)
		
		self.cnn_out_d = sample_output.shape[1]
		self.cnn_out_n = sample_output.shape[-1]

		if self.only_cnn: 
			self.fc_only_cnn = nn.Linear(self.cnn_out_d * self.cnn_out_n, 2)	
			if self.cnn_ext_out_d != -1:
				self.fc_ext_only_cnn = nn.Linear(self.cnn_ext_out_d * self.cnn_ext_out_n, 2)	
		else:
			if self.recurrent_type == 'lstm':
				self.lstm = nn.LSTM(self.cnn_out_d, self.hidden_size, num_layers=self.lstm_layers, bidirectional=True, batch_first=True)
				self.fc = nn.Linear(self.hidden_size*2, 2) #orig-5
			else:#transformer 
				self.pos_encoder = PositionalEncoding(d_model=self.cnn_out_d)
				#self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.cnn_out_d, nhead=nheads)
				self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.cnn_out_d, nhead=nheads), num_layers=nlayers)
				self.fc = nn.Linear(self.cnn_out_d * self.cnn_out_n, 2)
		
			if self.cnn_ext_out_d != -1:
				if self.recurrent_type == 'lstm':
					self.lstm_ext = nn.LSTM(self.cnn_ext_out_d, self.hidden_size, num_layers=self.lstm_layers, bidirectional=True, batch_first=True)
					self.fc_ext = nn.Linear(self.hidden_size*2, 2)
				else:
					self.pos_encoder_ext = PositionalEncoding(d_model=self.cnn_ext_out_d)
					#self.transformer_layer_ext = nn.TransformerEncoderLayer(d_model=self.cnn_ext_out_d, nhead=nheads)
					self.transformer_ext = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.cnn_ext_out_d, nhead=nheads), num_layers=nlayers)
					self.fc_ext = nn.Linear(self.cnn_ext_out_d * self.cnn_ext_out_n, 2)

			
		
		
		
	def forward(self, x): # x is a batch of audio samples in the time domain
		#preprocessing
		x = self.melspec(x)
		#print(x.shape)
		x = x + 1e-10
		x = x.log10()
		x = self.cmn(x)
		
		x = self.cnn(x)
		if self.cnn_type == 'vgg' or self.cnn_type == 'inceptionv3_s' or self.cnn_type == 'inceptionv3_m':
			if self.only_cnn:
				x = x.flatten(start_dim=1)
				x = self.fc_only_cnn(x)
				return x
			
			x = x.squeeze(2).permute(0,2,1)
			if self.recurrent_type == 'transformer':
				x = x * math.sqrt(self.cnn_out_d)
				x = self.pos_encoder(x)
				x = self.transformer(x)
				x = x.flatten(start_dim=1)
			else:
				output, (h_t, c_t) = self.lstm(x)
				x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)

			x = self.fc(x)
			return x
		else: #inceptionv3_l
			if self.training:
				x, aux = x
				if self.only_cnn:
					x, aux = x.flatten(start_dim=1), aux.flatten(start_dim=1)
					x, aux = self.fc_ext_only_cnn(x), self.fc_only_cnn(aux)
					return x, aux

				x = x.squeeze(2).permute(0,2,1)
				aux = aux.squeeze(2).permute(0,2,1)
				if self.recurrent_type == 'transformer':
					x = x * math.sqrt(self.cnn_ext_out_d)
					#x = self.pos_encoder_ext(x)
					x = self.transformer_ext(x)
					x = x.flatten(start_dim=1)

					aux = aux * math.sqrt(self.cnn_out_d)
					aux = self.pos_encoder(aux)
					aux = self.transformer(aux)
					aux = aux.flatten(start_dim=1)
				
				else:
					output, (h_t, c_t) = self.lstm_ext(x)
					x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)

					output, (h_t, c_t) = self.lstm(aux)
					aux = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)

				x, aux = self.fc_ext(x), self.fc(aux)
				return x, aux
			else:
				if self.only_cnn:
					x = x.flatten(start_dim=1)
					x = self.fc_ext_only_cnn(x)
					return x

				x = x.squeeze(2).permute(0,2,1)
				if self.recurrent_type == 'transformer':
					x = x * math.sqrt(self.cnn_ext_out_d)
					x = self.pos_encoder_ext(x)
					x = self.transformer_ext(x)
					x = x.flatten(start_dim=1)
				else:
					output, (h_t, c_t) = self.lstm_ext(x)
					x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)

				x = self.fc_ext(x)
				return x	
		
