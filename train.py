from model import CRNN
from dataloader import audio_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np



SEED = 42
torch.manual_seed(SEED)
#torch.backends.cudnn.benchmark=True


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyCollator(object):
	def __init__(self, num_frames):
		self.num_frames = num_frames
	def __call__(self, batch):
		label_list = []
		temp = []
		for spec, label in batch:
			cur_width = spec.shape[2]
			if cur_width < self.num_frames:
				temp.append(spec.repeat(1, 1, self.num_frames//cur_width + 1)[:, :, :self.num_frames])	
				label_list.append(label)
			else:
				num_multiples = cur_width // self.num_frames + 1
				req_width = num_multiples * self.num_frames
				new_spec = torch.cat( (spec, spec[:, :, :(req_width - cur_width)]), dim=2 )
				for i in range(num_multiples):
					temp.append(new_spec[:, :, i*self.num_frames:(i+1)*self.num_frames])
					label_list.append(label)
		specs = torch.stack(temp)
		labels = torch.tensor(label_list)
		return specs, labels	

parser = argparse.ArgumentParser(description='Training script for CRNN that performs LID')
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--beta1', help="Beta1 for Adam", type=float, default=0.9)
parser.add_argument('--beta2', help="Beta2 for Adam", type=float, default=0.999)
parser.add_argument('--batch-size', type=int, help='batch size', default=64)
parser.add_argument('--batch-size-val', type=int, help='batch size validation', default=64)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--num-frames', type=int, default=500)
parser.add_argument('--pix-per-sec', type=int, default=100)
parser.add_argument('--num-freq-levels', type=int, default=129)
parser.add_argument('--hidden-size', type=int, default=512)
args = parser.parse_args()


train_bs = args.batch_size
val_bs = args.batch_size_val


model = CRNN(hidden_size=args.hidden_size).double().to(device)

train_dataset = audio_dataset(root='./audio_data',csv_file='./audio2label_train.csv', pix_per_sec=args.pix_per_sec, num_freq_levels=args.num_freq_levels)
val_dataset = audio_dataset(root='./audio_data',csv_file='./audio2label_val.csv', pix_per_sec=args.pix_per_sec, num_freq_levels=args.num_freq_levels)

collater = MyCollator(args.num_frames)

train_loader = DataLoader(train_dataset, batch_size = train_bs, shuffle = True, collate_fn=collater)
val_loader = DataLoader(val_dataset, batch_size = val_bs, shuffle = True, collate_fn=collater)


criterion = nn.CrossEntropyLoss(reduction = 'sum')
optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
epochs = args.num_epochs
max_accuracy = -1
model_dir = './model_saves/'
best_epoch = 0
for epoch in range(epochs):
	torch.cuda.empty_cache()
	model.train()
	train_loss = 0
	train_correct_pred = 0
	for audio, label in tqdm(train_loader):
		audio, label = audio.double().to(device), label.to(device)
		pred = model(audio)
		loss = criterion(pred, label)
		pred_labels = torch.argmax(pred, dim = 1)
		train_correct_pred += torch.sum(pred_labels == label).item()
		train_loss += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
    
	print('Epoch: {} | Train Loss: {} | Train Accuracy: {}'.format(epoch, train_loss/len(train_dataset), train_correct_pred/len(train_dataset)))
	
	model.eval()
	with torch.no_grad():
		val_loss = 0
		val_correct_pred = 0
		conf_mat = np.zeros((5, 5))
		for audio, label in tqdm(val_loader):
			audio, label = audio.to(device), label.to(device)
			pred = model(audio)
			loss = criterion(pred, label)
			pred_labels = torch.argmax(pred, dim = 1)
			val_correct_pred += torch.sum(pred_labels == label).item()
			val_loss += loss.item()
		
			conf_mat += confusion_matrix(label.cpu(), pred_labels.cpu())
		
		val_accuracy = val_correct_pred/len(val_dataset)
		print('Val Loss: {} | Val Accuracy: {}'.format(val_loss/len(val_dataset), val_accuracy))
		print('****Confusion Matrix****')
		print(conf_mat)
		
		if val_accuracy > max_accuracy:
			max_accuracy = val_accuracy
			best_epoch = epoch
			print('Saving the model....')
			if not os.path.isdir(model_dir):
				os.makedirs(model_dir)
			torch.save(model, model_dir + str(epoch) + '.pth')

torch.save(model, model_dir + "final.pth")



 
            

