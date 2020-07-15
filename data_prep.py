import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import os
import sys

audio_length = 3.5
pre_emphasis = 0.97
frame_size = 0.025
frame_stride = 0.01
NFFT = 512

lang = sys.argv[1]
output_folder = 'audio_data_np'
root = 'audio_data'

files = os.listdir(lang)
for f in files:

    sample_rate, signal = scipy.io.wavfile.read(os.path.join(lang,f))  # File assumed to be in the same directory
    req_samples = int(sample_rate * audio_length)
    num_samples = len(signal)
    if num_samples < req_samples: #audio file too small
        signal = np.array(list(signal) * (req_samples//num_samples + 1))
        signal_modified = signal[:req_samples]
    else:
        extra = num_samples // req_samples + 1
        extra = extra*req_samples - num_samples
        signal_modified = np.concatenate((signal, signal[:extra]))
    # for i in range(    
    #     signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    for i in range(len(signal_modified)//req_samples):
        signal = signal_modified[req_samples * i : req_samples * (i + 1)]


        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        frames *= np.hamming(frame_length)
        # frames *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))  # Explicit Implementation **

        
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        # np.save(lang + "_" + str(j) + "_" + str(i) + '.npy', pow_frames)
        np.save(os.path.join(output_folder, lang) + '/' + os.path.splitext(f)[0] + '_' + str(i) + '.npy', pow_frames)
