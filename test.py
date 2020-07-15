from model import CRNN
from dataloader import audio_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_name = input("Enter model name")

test_dataset = audio_dataset(root='./audio_data_np',csv_file='./audio2label_test.csv')
test_bs = 5
test_loader = DataLoader(test_dataset, batch_size = test_bs, shuffle = True)

criterion = nn.CrossEntropyLoss(reduction = 'sum')

model = CRNN(hidden_size=512).double().to(device)
model = torch.load(model_name)

model.eval()
with torch.no_grad():
    test_loss = 0
    test_correct_pred = 0
    for audio, label in tqdm(test_loader):
        audio, label = audio.to(device), label.to(device)
        pred_labels = model(audio)
        loss = criterion(pred_labels, label)
        final_pred = torch.argmax(pred_labels, dim = 1)
        test_correct_pred += torch.sum(final_pred == label).item()
        test_loss += loss.item()
    
    test_accuracy = test_correct_pred/len(test_dataset)
    print('Test Loss: {} | Test Accuracy: {}'.format(test_loss/len(test_dataset), test_accuracy))