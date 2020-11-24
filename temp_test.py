import torch
from model import CRNN
from my_dataloader import dataloader_v2
from sklearn.metrics import confusion_matrix
from datetime import datetime as dt
import csv
import torchaudio
import math
from pprint import pprint


device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load('model_saves/vgg_lstm_subset/best.pth')
model = CRNN(hidden_size=checkpoint['hidden_size'], 
		only_cnn=checkpoint['only_cnn'], 
		cnn_type=checkpoint['cnn_type'], 
		recurrent_type=checkpoint['recurrent_type'],
		lstm_layers=checkpoint['lstm_layers'],
		nheads=checkpoint['nheads'], 
		nlayers=checkpoint['nlayers'],
		input_shape=checkpoint['input_shape']).double().to(device)
model.load_state_dict(checkpoint['model_state_dict'])


with open('./data/final_subset_shuffled_test.csv') as f:
	lines = f.readlines()
	data = [(line.strip().split(',')[0], int(line.strip().split(',')[1])) for line in lines]


model.eval()
with torch.no_grad():
	all_labels = []
	all_pred_labels = []
	small_audios_indices = []
	audio_lengths = []
	wrong_sample_rate_indices = []
	clean_indices = []
	total_mag = []
	#for batch_num, (audio, label) in enumerate(loader):
	for i, (path, label) in enumerate(data):
		audio, sample_rate = torchaudio.load('/storage' + path)
		if sample_rate != 8000:
			wrong_sample_rate_indices.append(i)
			continue
		if audio.shape[1] < 10100:
			#print("Audio at index {} is too small to pass through CNN".format(i))
			small_audios_indices.append(i)
			continue
		clean_indices.append(i)
		total_mag.append(torchaudio.transforms.Spectrogram(512, 200)(audio).sum())
		audio_lengths.append(audio.shape[1]/sample_rate)
		audio = audio.unsqueeze(0)
		audio = audio.double().to(device)
		pred = model(audio)
		pred_labels = torch.argmax(pred, dim=1)
		all_labels.append(label)
		all_pred_labels.append(pred_labels.cpu())
		if i % 1 == 0:
			print("[{}] On batch {}".format(dt.now(), i))


#all_labels = torch.cat(all_labels)
all_pred_labels = torch.cat(all_pred_labels)
all_labels = torch.tensor(all_labels)
audio_lengths = torch.tensor(audio_lengths)
clean_indices = torch.tensor(clean_indices)
total_mag = torch.stack(total_mag)

wrong_pred_indices = torch.where(all_pred_labels != all_labels)[0]

wrong_eng = []
for item in wrong_pred_indices:
	if all_pred_labels[item.item()] == 0:
		wrong_eng.append(item.item())

print(wrong_eng)
input()

orig_indices = clean_indices[wrong_pred_indices]
audio_lengths_wrong = audio_lengths[wrong_pred_indices]
total_mag_wrong = total_mag[wrong_pred_indices]

wrong_mags_and_indices = list(zip(total_mag_wrong.tolist(), orig_indices.tolist()))

low_mags = []
for item in wrong_mags_and_indices:
	if item[0] < 50:
		low_mags.append(item)
print(len(low_mags))
#input()
print(low_mags)

wrong_items_and_indices = list(zip(audio_lengths_wrong.tolist(), orig_indices.tolist()))
d = {}
for item in wrong_items_and_indices:
	if item[0] not in d:
		d[item[0]] = []
	d[item[0]].append(item[1])

pprint(d)
print(len(d))

d_repeats = {}
for item in d:
	if len(d[item]) > 1:
		d_repeats[item] = d[item]

pprint(d_repeats)
print(len(d_repeats))

for item in d_repeats:
	indices = d_repeats[item]
	for i in range(len(indices) - 1):
		first, _ = torchaudio.load('/storage' + data[indices[i]][0])
		for j in range(i+1, len(indices)):
			audio, _ = torchaudio.load('/storage' + data[indices[j]][0])
			result = audio.equal(first)
			if result:
				print(item, indices[i], indices[j], result)
			#input()

#print(audio_lengths_wrong, audio_lengths_wrong.shape)
print("Max: {} | Min: {} | Mean: {} | Median: {} | Std: {}".format(max(audio_lengths_wrong), min(audio_lengths_wrong), torch.mean(audio_lengths_wrong), torch.std(audio_lengths_wrong), torch.median(audio_lengths_wrong)))

#indices = torch.where(vad == 0)[0]
#all_labels_query = all_labels[indices]
#all_pred_labels_query = all_pred_labels[indices]

conf_mat = confusion_matrix(all_labels, all_pred_labels)
print(conf_mat)

print("Small audio indices: ", small_audios_indices) 
print("Wrong samplerate indicies: ", wrong_sample_rate_indices)

