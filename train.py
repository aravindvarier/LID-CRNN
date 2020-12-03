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
from my_dataloader import my_dataloader, dataloader_v2, dataloader_v3
from my_loss import focalLoss
import sys
from datetime import datetime as dt
from data_prep_scripts.freqs import freqs_func

torch.autograd.set_detect_anomaly(True)

SEED = 42
torch.manual_seed(SEED)
#torch.backends.cudnn.benchmark=True

#f = open('log.txt', 'w')
f = sys.stdout
pid = os.getpid()
print("PID: {}".format(pid), file=f, flush=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: {}".format(device), file=f, flush=True)

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
parser.add_argument('--lstm-layers', type=int, help='num layers in lstm', default=1)
parser.add_argument('--nheads', type=int, help='num heads in multi-head attention', default=4)
parser.add_argument('--nlayers', type=int, help='num layers in transformer', default=3)
parser.add_argument('--lr', type=float, help='learning rate', default=0.001)
parser.add_argument('--beta1', help="Beta1 for Adam", type=float, default=0.9)
parser.add_argument('--beta2', help="Beta2 for Adam", type=float, default=0.999)
parser.add_argument('--batch-size', type=int, help='batch size training', default=64)
parser.add_argument('--batch-size-val', type=int, help='batch size validation', default=64)
parser.add_argument('--num-epochs', type=int, help='number of epochs to train', default=30)
parser.add_argument('--hidden-size', type=int, help='hidden state size in lstm', default=512)
parser.add_argument('--only-cnn', type=str2bool, help='To use LSTM or not', choices=[False, True], default=False)
parser.add_argument('--exp-name', type=str, help='Experiment name used to create model save folder', default="")
parser.add_argument('--loader-type', type=str, help='Dataloder type. Default or Customv2/v3', default='v3', choices=['', 'v2', 'v3'])
parser.add_argument('--weighting-type', type=str, help='Weighting type for loss function', default='class-bal', choices=['inv-freq', 'class-bal', 'none'])
parser.add_argument('--num-seconds', type=float, help='Length of each audio file', default=10.0)
parser.add_argument('--N', type=int, help='total volume in class balanced weighting', default=1000000)
parser.add_argument('--csv', type=str, help='csv file to read files from', default='equal', choices=['subset', 'equal', 'full'])
parser.add_argument('--checkpoint-path', type=str, help='path to checkpoint to start training from', default='') 
parser.add_argument('--make-freqs', type=str2bool, help='Create freqs file or not', choices=[False, True], default=False)
parser.add_argument('--loss-type', type=str, help='cross entropy vs focal', choices=['ce','fl'], default='ce')
parser.add_argument('--gamma', type=float, help='gamma value for focal loss', default=2)
args = parser.parse_args()
args_dict = vars(args)

print(str(args_dict), file=f, flush=True)

train_bs = args.batch_size
val_bs = args.batch_size_val


if args.loader_type == "":
	train_dataset = audio_dataset(root='./data/spectrogram_data_fixed',csv_file='./data/audio2label_train.csv')
	print("Languages are indexed as: ", train_dataset.lang2id, file=f, flush=True)
	val_dataset = audio_dataset(root='./data/spectrogram_data_fixed',csv_file='./data/audio2label_val.csv')
	train_loader = DataLoader(train_dataset, batch_size = train_bs, shuffle = True, num_workers=4)
	val_loader = DataLoader(val_dataset, batch_size = val_bs, shuffle = True)
	criterion = nn.CrossEntropyLoss(reduction='sum')
else:
	if args.csv == 'full':
		csv = ''
	else:
		csv = '_' + args.csv

	if args.loader_type == 'v2':	
		train_loader = dataloader_v2('./data/final' + csv + '_shuffled_train.csv', batch_size=train_bs, num_seconds=args.num_seconds)
		val_loader = dataloader_v2('./data/final' + csv + '_shuffled_val.csv', batch_size=val_bs, num_seconds=args.num_seconds)
	else:
		train_loader = dataloader_v3('./data/final' + csv + '_shuffled_train.csv', batch_size=train_bs, num_seconds=args.num_seconds)
		val_loader = dataloader_v3('./data/final' + csv + '_shuffled_val.csv', batch_size=val_bs, num_seconds=args.num_seconds)

	if args.make_freqs:
		freqs_func('./data/final' + csv + '_shuffled_train.csv', './data/freqs' + csv + '.txt', 8000 * args.num_seconds, f)
	else:
		if not os.path.isfile('data/freqs' + csv + '.txt'):
			print('args.make-freqs set as False but no freqs file found. Ergo, creating freqs file anyway...', file=f, flush=True)
			freqs_func('./data/final' + csv + '_shuffled_train.csv', './data/freqs' + csv + '.txt', 8000 * args.num_seconds, f)
			
	lang2freq = {}
	ff = open('data/freqs' + csv + '.txt', 'r')
	for line in ff:
		lang, freq = line.strip().split()
		lang, freq = int(lang), int(freq)
		lang2freq[lang] = freq
	langs = sorted(lang2freq.keys())
	freqs = torch.tensor([lang2freq[lang] for lang in langs], dtype=torch.float64)

	#######testing#######
	freqs = torch.tensor([14000, 42000], dtype=torch.float64)
	####################

	freqs_copy = freqs.clone()
	print("Freqs: {}".format(freqs), file=f, flush=True)
	if args.weighting_type == 'none':
		weights = torch.tensor([1.2,0.8]).double()
	else:
		if args.weighting_type == 'class-bal':
			beta = (args.N - 1) / args.N
			freqs = (1 - beta**freqs) / (1 - beta) #effective number, class balanced loss [Yin Cui et al]
	
		weights = 1 / freqs
		weights = weights / weights.sum()
		weights = len(langs) * weights
		
	print("Weights: {}".format(weights), file=f, flush=True)
	weights = weights.to(device)		
	
	if args.loss_type == 'ce':
		criterion = nn.CrossEntropyLoss(weight=weights, reduction='sum')
	else:
		criterion = focalLoss(weight=weights, gamma=args.gamma, device=device, num_classes=len(langs))


#there has to be a better way to do this
#i'm doing this to get a sample audio to get the output shape from the neural network
for sample_audio, _ in train_loader:
	break

input_shape = sample_audio.shape

checkpoint_path = args.checkpoint_path
if checkpoint_path:
	checkpoint = torch.load(checkpoint_path)
	model = CRNN(hidden_size=checkpoint['hidden_size'], 
			only_cnn=checkpoint['only_cnn'], 
			cnn_type=checkpoint['cnn_type'], 
			recurrent_type=checkpoint['recurrent_type'],
			lstm_layers=checkpoint['lstm_layers'],
			nheads=checkpoint['nheads'], 
			nlayers=checkpoint['nlayers'],
			input_shape=checkpoint['input_shape']).double().to(device)
	model.load_state_dict(checkpoint['model_state_dict'])
	epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
else:
	model = CRNN(hidden_size=args.hidden_size, 
			only_cnn=args.only_cnn, 
			cnn_type=args.cnn_type, 
			recurrent_type=args.recurrent_type,
			lstm_layers=args.lstm_layers,
			nheads=args.nheads,
			nlayers=args.nlayers,
			input_shape=input_shape).double().to(device)
	epoch = 0


print("[{}] Number of trainable parameters are: {}".format(dt.now(), count_parameters(model)), file=f, flush=True)

#total = 0
#for p in model.named_parameters():
#	if p[1].requires_grad:
#		if 'cnn' in p[0]:
#			continue
#		else:
#			total += p[1].numel()
#			print(p[0], p[1].numel())
#			input()
#print(total)
#exit()

#writing this function just in case in future i might want to use it
def adjust_optim(optimizer):
	for param_group in optimizer.param_groups():
		param_group['lr'] -= 0.0001


optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
epochs = args.num_epochs
max_loss = 2**32
best_recall = 0
best_f1 = 0

if not args.exp_name:
	exp_name = args.cnn_type + "_" + str(args.only_cnn) + "_" + ("" if args.only_cnn else str(args.hidden_size))
else:
	exp_name = args.exp_name
model_dir = 'model_saves/' + exp_name


best_epoch = 0 
while epoch <= epochs:
	torch.manual_seed(SEED + epoch)
	torch.cuda.empty_cache()
	model.train()
	model.to(device)
	train_loss = 0
	train_loss_aux = 0
	total_train_correct_pred = 0
	total_train_correct_pred_aux = 0
	for batch_num, (audio, label) in enumerate(train_loader): #enumerate(tqdm(train_loader, total=int(freqs_copy.sum()/train_bs)+1)):
		audio, label = audio.double().to(device), label.to(device)
		if args.cnn_type == 'vgg' or args.cnn_type == 'inceptionv3_s' or args.cnn_type == 'inceptionv3_m':
			train_correct_pred = 0
			pred = model(audio)
			loss = criterion(pred, label)
			pred_labels = torch.argmax(pred, dim = 1)
			train_correct_pred = torch.sum(pred_labels == label).item()
			total_train_correct_pred += train_correct_pred 
			train_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if batch_num % 10 == 0:
				conf_mat = confusion_matrix(label.cpu(), pred_labels.cpu(), labels=langs)
				fl_recall = conf_mat[0][0]/conf_mat[0].sum() if conf_mat[0].sum() else 0
				print("**Batch confusion matrix**", file=f, flush=True)
				print(conf_mat, file=f, flush=True)
				print("[{}] Batch:{} Accuracy(current batch): {} | FL Recall: {}".format(dt.now(), batch_num, train_correct_pred/len(label), fl_recall), file=f, flush=True)
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
			if batch_num % 10 == 0:
				conf_mat = confusion_matrix(label.cpu(), pred_labels.cpu(), labels=langs)
				fl_recall = conf_mat[0][0]/conf_mat[0].sum() if conf_mat[0].sum() else 0
				print("**Batch confusion matrix**", file=f, flush=True)
				print(conf_mat)
				conf_mat_aux = confusion_matrix(label.cpu(), pred_labels_aux.cpu(), labels=langs)
				fl_recall_aux = conf_mat_aux[0][0]/conf_mat_aux[0].sum() if conf_mat_aux[0].sum() else 0
				print("**Batch confusion matrix aux**", file=f, flush=True)
				print(conf_mat_aux)
				print("[{}] Batch:{} Accuracy main: {} | Accuracy aux: {}".format(dt.now(), batch_num, train_correct_pred/len(label), train_correct_pred_aux/len(label)), file=f, flush=True)
		
			


	if args.cnn_type == 'vgg' or args.cnn_type == 'inceptionv3_s' or args.cnn_type == 'inceptionv3_m':
		if args.loader_type == "":
			print('[{}] Epoch: {} | Train Loss: {} | Train Accuracy: {}'.format(dt.now(),
											epoch, 
											train_loss/len(train_dataset), 
											total_train_correct_pred/len(train_dataset)), file=f, flush=True)
		elif args.loader_type == 'v3':
			print('[{}] Epoch: {} | Train Loss: {} | Train Accuracy: {}'.format(dt.now(),
											epoch, 
											train_loss/len(train_loader), 
											total_train_correct_pred/len(train_loader)), file=f, flush=True)

		else:
			print('[{}] Epoch: {} | Train Loss: {} | Train Accuracy: {}'.format(dt.now(), 
											epoch, 
											train_loss/train_loader.total_samples, 
											total_train_correct_pred/train_loader.total_samples), file=f, flush=True)
	else:
		if args.loader_type == "":
			print('[{}] Epoch: {} | Train Loss: {} | Train Loss(Aux): {} | Train Accuracy: {} | Train Accuracy(Aux): {} '.format(dt.now(), 
																	epoch, 
																	train_loss/len(train_dataset), 
																	train_loss_aux/len(train_dataset), 
																	total_train_correct_pred/len(train_dataset), 
																	total_train_correct_pred_aux/len(train_dataset)), file=f, flush=True)
		elif args.loader_type == 'v3':
			print('[{}] Epoch: {} | Train Loss: {} | Train Loss(Aux): {} | Train Accuracy: {} | Train Accuracy(Aux): {} '.format(dt.now(), 
																	epoch, 
																	train_loss/len(train_loader), 
																	train_loss_aux/len(train_loader), 
																	total_train_correct_pred/len(train_loader), 
																	total_train_correct_pred_aux/len(train_loader)), file=f, flush=True)
		else:
			print('[{}] Epoch: {} | Train Loss: {} | Train Loss(Aux): {} | Train Accuracy: {} | Train Accuracy(Aux): {} '.format(dt.now(), 
																	epoch, 
																	train_loss/train_loader.total_samples, 
																	train_loss_aux/train_loader.total_samples, 
																	total_train_correct_pred/train_loader.total_samples, 
																	total_train_correct_pred_aux/train_loader.total_samples), file=f, flush=True)

	
	
	model.eval()
	with torch.no_grad():
		val_loss = 0
		val_correct_pred = 0
		all_labels = []
		all_pred_labels = []
		for audio, label in val_loader: #tqdm(val_loader):
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
		
		fl_recall = conf_mat[0][0]/conf_mat[0].sum() if conf_mat[0][0] else 0
		fl_precision = conf_mat[0][0]/conf_mat[:,0].sum() if conf_mat[0][0] else 0
		fl_f1 = 0
		if fl_recall + fl_precision:
			fl_f1 = 2 * fl_recall * fl_precision / (fl_recall + fl_precision)
		if args.loader_type == "":
			val_accuracy = val_correct_pred/len(val_dataset)
			val_loss_avg = val_loss/len(val_dataset)
			print('[{}] Epoch: {} | Val Loss: {} | Val Accuracy: {} | FL recall: {} | FL precision: {} | FL F1: {}'.format(dt.now(), 
													epoch, 
													val_loss_avg, 
													val_accuracy,
													fl_recall,
													fl_precision,
													fl_f1), file=f, flush=True)
		elif args.loader_type == 'v3':
			val_accuracy = val_correct_pred/len(val_loader)
			val_loss_avg = val_loss/len(val_loader)
			print('[{}] Epoch: {} | Val Loss: {} | Val Accuracy: {} | FL recall: {}, FL precision: {} | FL F1: {}'.format(dt.now(), 
													epoch, 
													val_loss_avg, 
													val_accuracy,
													fl_recall,
													fl_precision,
													fl_f1), file=f, flush=True)
		else:
			val_accuracy = val_correct_pred/len(all_labels)
			val_loss_avg = val_loss/len(all_labels)
			print('[{}] Epoch: {} | Val Loss: {} | Val Accuracy: {} | FL recall: {}, FL precision: {} | FL F1: {}'.format(dt.now(), 
													epoch, 
													val_loss_avg, 
													val_accuracy, 
													fl_recall,
													fl_precision,
													fl_f1), file=f, flush=True)
		print('****Confusion Matrix****', file=f, flush=True)
		print(conf_mat, file=f, flush=True)
		
		#Early stopping criteria is validation loss. (Some resources say to use Validation accuracy. I think loss makes more sense)
		if val_loss_avg < max_loss:
			max_loss = val_loss_avg
			best_epoch = epoch
			print('Saving the model for better loss...', file=f, flush=True)
			if not os.path.isdir(model_dir):
				os.makedirs(model_dir)

			torch.save({'model_state_dict' : model.state_dict(), 
					'hidden_size' : args.hidden_size, 
					'only_cnn' : args.only_cnn, 
					'cnn_type' : args.cnn_type, 
					'recurrent_type' : args.recurrent_type,
					'lstm_layers' : args.lstm_layers,
					'nheads' : args.nheads, 
					'nlayers' : args.nlayers,
					'input_shape' : input_shape,
					'epoch' : epoch}, os.path.join(model_dir ,'best.pth'))

		if fl_recall > best_recall:
			best_recall = fl_recall
			print('Saving the model for better recall...', file=f, flush=True)
			if not os.path.isdir(model_dir):
				os.makedirs(model_dir)

			torch.save({'model_state_dict' : model.state_dict(), 
					'hidden_size' : args.hidden_size, 
					'only_cnn' : args.only_cnn, 
					'cnn_type' : args.cnn_type, 
					'recurrent_type' : args.recurrent_type,
					'lstm_layers' : args.lstm_layers,
					'nheads' : args.nheads, 
					'nlayers' : args.nlayers,
					'input_shape' : input_shape,
					'epoch' : epoch}, os.path.join(model_dir ,'best_recall.pth'))
		if fl_f1 > best_f1:
			best_f1 = fl_f1
			print('Saving the model for better f1...', file=f, flush=True)
			if not os.path.isdir(model_dir):
				os.makedirs(model_dir)

			torch.save({'model_state_dict' : model.state_dict(), 
					'hidden_size' : args.hidden_size, 
					'only_cnn' : args.only_cnn, 
					'cnn_type' : args.cnn_type, 
					'recurrent_type' : args.recurrent_type,
					'lstm_layers' : args.lstm_layers,
					'nheads' : args.nheads, 
					'nlayers' : args.nlayers,
					'input_shape' : input_shape,
					'epoch' : epoch}, os.path.join(model_dir ,'best_f1.pth'))



	print("Saving latest model...", file=f, flush=True)
	torch.save({'model_state_dict' : model.state_dict(), 
			'hidden_size' : args.hidden_size, 
			'only_cnn' : args.only_cnn, 
			'cnn_type' : args.cnn_type,
			'recurrent_type' : args.recurrent_type,
			'lstm_layers' : args.lstm_layers,
			'nheads' : args.nheads, 
			'nlayers' : args.nlayers,
			'input_shape' : input_shape,
			'epoch' : epoch}, os.path.join(model_dir ,'latest.pth'))


	epoch += 1

args_writer = open(model_dir+'/args.txt', 'w')
args_writer.write(str(args_dict))
args_writer.close()


 
            

