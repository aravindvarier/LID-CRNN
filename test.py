from model import CRNN
from dataloader import audio_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_path = input("Enter path to model: ")


parser = argparse.ArgumentParser(description='Testing script for CRNN that performs LID')
parser.add_argument('--batch-size-test', type=int, help='batch size testing', default=64)
args = parser.parse_args()


test_dataset = audio_dataset(root='./data/spectrogram_data_fixed',csv_file='./data/audio2label_test.csv')
test_bs = args.batch_size_test
test_loader = DataLoader(test_dataset, batch_size = test_bs, shuffle = True)

criterion = nn.CrossEntropyLoss(reduction = 'sum')

checkpoint = torch.load(model_path)
model = CRNN(hidden_size=checkpoint['hidden_size']).double().to(device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
with torch.no_grad():
	test_loss = 0
	test_correct_pred = 0
	conf_mat = np.zeros((5, 5))
	for audio, label in tqdm(test_loader):
		audio, label = audio.double().to(device), label.to(device)
		pred = model(audio)
		loss = criterion(pred, label)
		pred_labels = torch.argmax(pred, dim = 1)
		test_correct_pred += torch.sum(pred_labels == label).item()
		test_loss += loss.item()

		conf_mat += confusion_matrix(label.cpu(), pred_labels.cpu())


test_accuracy = test_correct_pred/len(test_dataset)
print('Test Loss: {} | Test Accuracy: {}'.format(test_loss/len(test_dataset), test_accuracy))
print('****Confusion Matrix****')
print(conf_mat)

