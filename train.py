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

torch.autograd.set_detect_anomaly(True)

SEED = 42
torch.manual_seed(SEED)
#torch.backends.cudnn.benchmark=True


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)	

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Training script for CRNN that performs LID', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cnn-type', type=str, help='Type of CNN used', default='vgg', choices=['vgg', 'inceptionv3_l', 'inceptionv3_m', 'inceptionv3_s'])
parser.add_argument('--recurrent-type', type=str, help='Type of Recurrent network used', default='lstm', choices=['lstm', 'transformer'])
parser.add_argument('--nheads', type=int, help='num heads in multi-head attention', default=4)
parser.add_argument('--nlayers', type=int, help='num layers in transformer', default=3)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--beta1', help="Beta1 for Adam", type=float, default=0.9)
parser.add_argument('--beta2', help="Beta2 for Adam", type=float, default=0.999)
parser.add_argument('--batch-size', type=int, help='batch size training', default=64)
parser.add_argument('--batch-size-val', type=int, help='batch size validation', default=64)
parser.add_argument('--num-epochs', type=int, help='number of epochs to train', default=100)
parser.add_argument('--hidden-size', type=int, help='hidden state size in lstm', default=512)
parser.add_argument('--only-cnn', type=str2bool, help='To use LSTM or not', choices=[False, True], default=False)
parser.add_argument('--exp-name', type=str, help='Experiment name used to create model save folder', default = "")
args = parser.parse_args()
args_dict = vars(args)




train_bs = args.batch_size
val_bs = args.batch_size_val


model = CRNN(hidden_size=args.hidden_size, 
			only_cnn=args.only_cnn, 
			cnn_type=args.cnn_type, 
			recurrent_type=args.recurrent_type,
			nheads=args.nheads,
			nlayers=args.nlayers).double().to(device)
print("Number of trainable parameters are: {}".format(count_parameters(model)))

train_dataset = audio_dataset(root='./data/spectrogram_data_fixed',csv_file='./data/audio2label_train.csv')
print("Languages are indexed as: ", train_dataset.lang2id)
val_dataset = audio_dataset(root='./data/spectrogram_data_fixed',csv_file='./data/audio2label_val.csv')


train_loader = DataLoader(train_dataset, batch_size = train_bs, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = val_bs, shuffle = True)


criterion = nn.CrossEntropyLoss(reduction = 'sum')
optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = (args.beta1, args.beta2))
epochs = args.num_epochs
max_accuracy = -1

if not args.exp_name:
	exp_name = args.cnn_type + "_" + str(args.only_cnn) + "_" + ("" if args.only_cnn else str(args.hidden_size))
else:
	exp_name = args.exp_name
model_dir = 'model_saves/' + exp_name


best_epoch = 0
for epoch in range(epochs):
	torch.cuda.empty_cache()
	model.train()
	train_loss = 0
	train_loss_aux = 0
	total_train_correct_pred = 0
	total_train_correct_pred_aux = 0
	for batch_num, (audio, label) in enumerate(tqdm(train_loader)):
		audio, label = audio.double().to(device), label.to(device)
		if args.cnn_type == 'vgg' or args.cnn_type == 'inceptionv3_s' or args.cnn_type == 'inceptionv3_m':
			pred = model(audio)
			loss = criterion(pred, label)
			pred_labels = torch.argmax(pred, dim = 1)
			total_train_correct_pred += torch.sum(pred_labels == label).item()
			train_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		else: #inception_large
			train_correct_pred = train_correct_pred_aux = 0
			pred, pred_aux = model(audio)
			loss = criterion(pred, label)
			loss_aux = criterion(pred_aux, label)
			pred_labels = torch.argmax(pred, dim=1)
			pred_labels_aux = torch.argmax(pred_aux, dim=1)
			train_correct_pred = torch.sum(pred_labels == label).item()
			total_train_correct_pred += train_correct_pred
			train_correct_pred_aux = torch.sum(pred_labels_aux == label).item()
			total_train_correct_pred_aux += train_correct_pred_aux 
			train_loss += loss.item()
			train_loss_aux += loss_aux.item()
			optimizer.zero_grad()
			total_loss = loss + loss_aux
			total_loss.backward()
			optimizer.step()
			if batch_num % 50 == 0:
				print("Accuracy main: {} | Accuracy aux: {}".format(train_correct_pred/len(label), train_correct_pred_aux/len(label)))
		
			


	if args.cnn_type == 'vgg' or args.cnn_type == 'inceptionv3_s' or args.cnn_type == 'inceptionv3_m':
		print('Epoch: {} | Train Loss: {} | Train Accuracy: {}'.format(epoch, train_loss/len(train_dataset), total_train_correct_pred/len(train_dataset)))
	else:
		print('Epoch: {} | Train Loss: {} | Train Loss(Aux): {} | Train Accuracy: {} | Train Accuracy(Aux): {} '.format(epoch, train_loss/len(train_dataset), train_loss_aux/len(train_dataset), total_train_correct_pred/len(train_dataset), total_train_correct_pred_aux/len(train_dataset)))

	
	
	model.eval()
	with torch.no_grad():
		val_loss = 0
		val_correct_pred = 0
		all_labels = []
		all_pred_labels = []
		for audio, label in tqdm(val_loader):
			audio, label = audio.double().to(device), label.to(device)
			pred = model(audio)
			loss = criterion(pred, label)
			pred_labels = torch.argmax(pred, dim = 1)
			all_labels.append(label.cpu())
			all_pred_labels.append(pred_labels.cpu())
			val_correct_pred += torch.sum(pred_labels == label).item()
			val_loss += loss.item()
		all_labels = torch.cat(all_labels)
		all_pred_labels = torch.cat(all_pred_labels)
		conf_mat = confusion_matrix(all_labels, all_pred_labels)
		
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

			torch.save({'model_state_dict' : model.state_dict(), 'hidden_size' : args.hidden_size, 'only_cnn' : args.only_cnn, 'cnn_type' : args.cnn_type, 'recurrent_type' : args.recurrent_type, 'nheads' : args.nheads, 'nlayers' : args.nlayers}, os.path.join(model_dir ,'best.pth'))
		


print("Saving final model...")
torch.save({'model_state_dict' : model.state_dict(), 'hidden_size' : args.hidden_size, 'only_cnn' : args.only_cnn, 'cnn_type' : args.cnn_type}, os.path.join(model_dir ,'final.pth'))

args_writer = open(model_dir+'/args.txt', 'w')
args_writer.write(vars(args))
args_writer.close()


 
            

