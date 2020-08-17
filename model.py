import torch.nn as nn
import torch
import numpy as np
#from torchvision import models
from my_inception import Inception3_small, Inception3_medium, Inception3_large

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class VGG(nn.Module):
	def __init__(self):
		super(VGG, self).__init__()
		self.cnn = nn.Sequential(nn.Conv2d(1,16, (7,7)),
								nn.ReLU(),
								nn.BatchNorm2d(16),
								nn.MaxPool2d((2,2), 2),
							#	nn.Dropout(p=0.1),
                               	
								nn.Conv2d(16, 32, (5,5)),
								nn.ReLU(),
								nn.BatchNorm2d(32),
								nn.MaxPool2d((2,2), 2),
							#	nn.Dropout(p=0.1),

								nn.Conv2d(32,64,(3,3)),
								nn.ReLU(),
								nn.BatchNorm2d(64),
								nn.MaxPool2d((2,2), 2),
							#	nn.Dropout(p=0.1),
				
								nn.Conv2d(64,128,(3,3)),
								nn.ReLU(),
								nn.BatchNorm2d(128),
								nn.MaxPool2d((2,2), 2),
							#	nn.Dropout(p=0.1),
                                
								nn.Conv2d(128,256, (3,3)),
                                nn.ReLU(),
                                nn.BatchNorm2d(256),
                                nn.MaxPool2d((2,2), 2),
							#	nn.Dropout(p=0.1),
								)
	def forward(self, x):
		return self.cnn(x)

#class InceptionV3(nn.Module):
#	def __init__(self):
#		super(InceptionV3, self).__init__()
#		self.cnn = Inception3_large()
#		#self.cnn.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
#		#self.cnn.AuxLogits.fc = Identity()
#		#self.cnn.fc = Identity()
#
#	def forward(self, x):
#		return self.cnn(x)

class CRNN(nn.Module):
	def __init__(self, hidden_size, cnn_type, only_cnn=False):
		super(CRNN, self).__init__()
		self.hidden_size = hidden_size
		self.only_cnn = only_cnn
		self.cnn_type = cnn_type
		
		if self.cnn_type == 'vgg':
			self.cnn = VGG()
		elif self.cnn_type == 'inceptionv3_l':
			self.cnn = Inception3_large()
		elif self.cnn_type == 'inceptionv3_m':
			self.cnn = Inception3_medium()
		else:
			self.cnn = Inception3_small()
		
		
		self.lstm = nn.LSTM(256, self.hidden_size, bidirectional = True, batch_first = True)
		self.fc_vgg = nn.Linear(self.hidden_size*2, 5)
		self.fc_vgg_only_cnn = nn.Linear(256 * 13, 5)

		#LSTMs and FCs for Inception
		self.lstm_main = nn.LSTM(2048, self.hidden_size, bidirectional=True, batch_first=True)
		self.lstm_aux = nn.LSTM(128, self.hidden_size, bidirectional=True, batch_first=True)

		self.fc_inception = nn.Linear(self.hidden_size*2, 5)
		self.fc_inception_aux = nn.Linear(self.hidden_size*2, 5)
		self.fc_inception_only_cnn = nn.Linear(2048*14, 5)
		self.fc_inception_aux_only_cnn = nn.Linear(128*9, 5)
	
	def forward(self, x):
		x = self.cnn(x)
		if self.cnn_type == 'vgg':
			if self.only_cnn:
				x = x.flatten(start_dim=1)
				x = self.fc_vgg_only_cnn(x)
				return x
			x = x.squeeze(2).permute(0,2,1)
			output, (h_t, c_t) = self.lstm(x)
			x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)
			x = self.fc_vgg(x)
			return x
		elif self.cnn_type == 'inceptionv3_l':
			if self.training:
				x, aux = x
				if self.only_cnn:
					x, aux = x.flatten(start_dim=1), aux.flatten(start_dim=1)
					x, aux = self.fc_inception_only_cnn(x), self.fc_inception_aux_only_cnn(aux)
					return x, aux
				x = x.squeeze(2).permute(0,2,1)
				output, (h_t, c_t) = self.lstm_main(x)
				x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)
				x = self.fc_inception(x)

				aux = aux.squeeze(2).permute(0,2,1)
				output, (h_t, c_t) = self.lstm_aux(aux)
				aux = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)
				aux = self.fc_inception_aux(aux)
				return x, aux
			else:
				if self.only_cnn:
					x = x.flatten(start_dim=1)
					x = self.fc_inception_only_cnn(x)
					return x
				x = x.squeeze(2).permute(0,2,1)
				output, (h_t, c_t) = self.lstm_main(x)
				x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)
				x = self.fc_inception(x)
				return x	
		else: #Either inception_small or inception_medium
			if self.only_cnn:
				x = x.flatten(start_dim=1)
				x = self.fc_inception_aux_only_cnn(x)
				return x
			x = x.squeeze(2).permute(0,2,1)
			output, (h_t, c_t) = self.lstm_aux(x)
			x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)
			x = self.fc_inception_aux(x)
			return x

		#if self.only_cnn:
		#	x = x.flatten(start_dim=1)
		#	if self.cnn_type == 'vgg':
		#		x = self.fc_vgg_only_cnn(x)
		#	else:
		#		x = self.fc_inception_aux_only_cnn(x)
		#	return x

		#x = x.squeeze(2).permute(0,2,1)
		#if self.cnn_type == 'vgg':
		#	output, (h_t, c_t) = self.lstm(x)
		#	x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)
		#	x = self.fc_vgg(x)
		#else:
		#	output, (h_t, c_t) = self.lstm_aux(x)
		#	x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)
		#	x = self.fc_inception_aux(x)
		#return x


