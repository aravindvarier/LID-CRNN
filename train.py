from model import CRNN
from dataloader import audio_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix


device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_bs = 64
val_bs = 64
test_bs = 64


model = CRNN(hidden_size=512).double().to(device)

train_dataset = audio_dataset(root='./audio_data_np',csv_file='./audio2label_train.csv')
val_dataset = audio_dataset(root='./audio_data_np',csv_file='./audio2label_val.csv')

train_loader = DataLoader(train_dataset, batch_size = train_bs, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = val_bs, shuffle = True)


criterion = nn.CrossEntropyLoss(reduction = 'sum')
optimizer = optim.Adam(model.parameters())
epochs = 100
max_accuracy = -1
model_dir = './model_saves/'
best_epoch = 0
for epoch in range(epochs):
    torch.cuda.empty_cache()
    model.train()
    train_loss = 0
    train_correct_pred = 0
    for audio, label in tqdm(train_loader):
        audio, label = audio.to(device), label.to(device)
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



 
            

