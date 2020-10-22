import os
import argparse
import csv

csv_file = open('data/final_equal.csv', 'r')
csv_reader = csv.reader(csv_file, delimiter=',')

#langs = os.listdir(src_dir)
#lang2id = {lang: i for i,lang in enumerate(langs)}
int2lang = {'0':'fl', '1':'eng'}

dst_dir = 'data/spectrogram_data'
if not os.path.isdir(dst_dir):
	os.mkdir(dst_dir)

parser = argparse.ArgumentParser(description='Script which uses the sox tool to convert audio to image spectrograms')
parser.add_argument('--pix-per-sec', type=int, default=100)
parser.add_argument('--num-freq-levels', type=int, default=129)
parser.add_argument('--print-freq', type=int, default=500)
args = parser.parse_args()



#for lang in langs:
#	print("Converting from wav to png for language: {}".format(lang))
#	files = os.listdir(os.path.join(src_dir,lang))
#	if not os.path.isdir(os.path.join(dst_dir, lang)):
#		os.mkdir(os.path.join(dst_dir, lang))
#	for i, f in enumerate(files):
#		wav_file = os.path.join(src_dir, lang, f)
#		png_file = os.path.join(dst_dir, lang, os.path.splitext(f)[0] + ".png")		
#		command = "sox -V0 {} -n remix 1 rate 8k spectrogram -y {} -X {} -m -r -o {}".format(wav_file, args.num_freq_levels, args.pix_per_sec, png_file)
#		os.system(command)
#		if i % args.print_freq == 0:
#			print("Completed {} conversions".format(i))



for i, line in enumerate(csv_reader):
	audio_file = '/storage' + line[0]
	lang = int2lang[line[1]]
	if not os.path.isdir(os.path.join(dst_dir, lang)):
		os.mkdir(os.path.join(dst_dir, lang))
	parts = audio_file.split('/')
	png_file = parts[-2] + "_" + os.path.splitext(parts[-1])[0] + ".png"
	png_file = os.path.join(dst_dir, lang, png_file)
	command = "sox -V0 {} -n remix 1 rate 8k spectrogram -y {} -X {} -m -r -o {}".format(audio_file, args.num_freq_levels, args.pix_per_sec, png_file)
	os.system(command)
	if i % args.print_freq == 0:
		print("Completed {} converesions".format(i))
