import os
import argparse
import torch
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser(description='Breaks up png images to equal sizes')
parser.add_argument('--num-frames', type=int, default=500)
parser.add_argument('--print-freq', type=int, default=500)
args = parser.parse_args()


num_frames = args.num_frames

src_dir = 'data/spectrogram_data'
langs = os.listdir(src_dir)
lang2id = {lang: i for i,lang in enumerate(langs)}

dst_dir = 'data/spectrogram_data_fixed'
if not os.path.isdir(dst_dir):
	os.mkdir(dst_dir)

img2tensor = transforms.ToTensor()
tensor2img = transforms.ToPILImage()

for lang in langs:
	print("Fixing language: {}".format(lang))
	files = os.listdir(os.path.join(src_dir,lang))
	if not os.path.isdir(os.path.join(dst_dir, lang)):
		os.mkdir(os.path.join(dst_dir, lang))
	for n, f in enumerate(files):
		name, ext = os.path.splitext(f)
		img_file_name = os.path.join(src_dir, lang, f)
		spec = Image.open(img_file_name)
		spec = img2tensor(spec)
		dst_file_name = os.path.join(dst_dir, lang, f)

		cur_width = spec.shape[2]
		if cur_width < num_frames:
			new_spec = spec.repeat(1, 1, num_frames//cur_width + 1)[:, :, :num_frames]	
			dst_img = tensor2img(new_spec)
			dst_img.save(dst_file_name)
			
		else:
			num_multiples = cur_width // num_frames + 1
			req_width = num_multiples * num_frames
			new_spec = torch.cat( (spec, spec[:, :, :(req_width - cur_width)]), dim=2 )
			for i in range(num_multiples):
				dst_file_name = os.path.join(dst_dir, lang, name + "." + str(i) + ext)
				dst_img = new_spec[:, :, i*num_frames:(i+1)*num_frames]
				dst_img = tensor2img(dst_img)
				dst_img.save(dst_file_name)


		if n % args.print_freq == 0:
			print("Finished fixing {} image".format(n))
