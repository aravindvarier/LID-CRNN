import torch
import argparse
import torchaudio
from model import CRNN
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='LID testing script for single audio')
parser.add_argument('--input', type=str, help='path to input audio', required=True)
parser.add_argument('--output', type=str, help='path to output result file', default='./result.txt')
args = parser.parse_args()

out_f = open(args.output, 'w')


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
model.eval()

audio, sample_rate = torchaudio.load(args.input)
assert(sample_rate == 8000)

audio = audio.unsqueeze(0)
audio = audio.double().to(device)
with torch.no_grad():
	pred = model(audio)
probs = F.softmax(pred, dim=1)
pred_label = torch.argmax(pred, dim=1)


out_f.write("Labels- {0: FL, 1: EN}\n")
out_f.write("Model prediction in probabilites: {}\n".format(probs.cpu().tolist()))
out_f.write("Predicted label: {}\n".format(pred_label.cpu().item()))


