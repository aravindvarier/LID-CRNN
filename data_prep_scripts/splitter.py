import csv
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Script which splits csv file into Train, Val and Test csv files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('file_name', type=str)
args = parser.parse_args()

name = args.file_name.split('/')[-1].split('.')[0]

df = pd.read_csv(args.file_name, header=None)
fl_df = df[df.iloc[:, 1] == 0]
fl_df = fl_df.sample(frac=1).reset_index(drop=True)
eng_df = df[df.iloc[:, 1] == 1]
eng_df = eng_df.sample(frac=1).reset_index(drop=True)

train_df = pd.concat([fl_df.iloc[:int(0.8*len(fl_df))], eng_df.iloc[:int(0.8*len(eng_df))]])
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.to_csv('data/'+name+'_shuffled_train.csv', header=False, index=False)

val_df = pd.concat([fl_df.iloc[int(0.8*len(fl_df)):int(0.9*len(fl_df))], eng_df.iloc[int(0.8*len(eng_df)):int(0.9*len(eng_df))]])
val_df = val_df.sample(frac=1).reset_index(drop=True)
val_df.to_csv('data/'+name+'_shuffled_val.csv', header=False, index=False)

test_df = pd.concat([fl_df.iloc[int(0.9*len(fl_df)):], eng_df.iloc[int(0.9*len(eng_df)):]])
test_df = test_df.sample(frac=1).reset_index(drop=True)
test_df.to_csv('data/'+name+'_shuffled_test.csv', header=False, index=False)

