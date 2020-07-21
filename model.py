import torch.nn as nn
import torch
import numpy as np
from torchvision import models



class CRNN(nn.Module):
	def __init__(self, hidden_size):
		super(CRNN, self).__init__()
		self.hidden_size = hidden_size
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

							#	nn.Conv2d(256, 512, (3,3)),
							#	nn.ReLU(),
							#	nn.BatchNorm2d(512),
							#	nn.MaxPool2d((2,2), 2),
								)
		self.lstm = nn.LSTM(256, self.hidden_size, bidirectional = True, batch_first = True)
		self.fc = nn.Linear(self.hidden_size*2, 5)
	
	def forward(self, x):
		x = self.cnn(x)
		x = x.squeeze(2).permute(0,2,1)
		output, (h_t, c_t) = self.lstm(x)
		x = torch.cat((output[:, -1, :self.hidden_size//2], output[:, 0, self.hidden_size//2:]), dim = 1)
		x = self.fc(x)
		return(x)

