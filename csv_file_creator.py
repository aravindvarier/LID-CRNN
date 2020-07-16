import csv
from random import shuffle
import os 

train_file = "audio2label_train.csv"
val_file = "audio2label_val.csv"
test_file = "audio2label_test.csv"

root = 'audio_data_np'
langs = os.listdir(root)
lang2id = {lang: i for i,lang in enumerate(langs)}
print("Languages are indexed as: ", lang2id)



for lang in langs:
    files = os.listdir(os.path.join(root,lang))
    shuffle(files)

    with open(train_file, 'a') as a2l:
        audio_writer = csv.writer(a2l, delimiter = ',')    
        for f in files[:int(0.8 * len(files))]:    
            audio_writer.writerow([os.path.join(root,lang,f), lang2id[lang]])

    with open(val_file, 'a') as a2l:
        audio_writer = csv.writer(a2l, delimiter = ',')    
        for f in files[int(0.8 * len(files)) : int(0.9 * len(files))]:    
            audio_writer.writerow([os.path.join(root,lang,f), lang2id[lang]])

    with open(test_file, 'a') as a2l:
        audio_writer = csv.writer(a2l, delimiter = ',')    
        for f in files[int(0.9 * len(files)):]:    
            audio_writer.writerow([os.path.join(root,lang,f), lang2id[lang]])