# Code for evaluating piano to guitar sound transfer
# Name the files as below
# pred_1.wav, tgt_1.wav, etc.
#
# python evaluate.py -pred <prediction_dir> -tgt <target_dir>

import os
import random
import numpy as np
from PIL import Image

# Audio processing stuff
import librosa
import librosa.display
import scipy
from scipy.io import wavfile # get the api
from matplotlib import pyplot as plt
import soundfile as sf
import argparse

# Parse args
parser = argparse.ArgumentParser(description='Evaluate piano to guitar transfer.')
parser.add_argument('-pred', dest='pred', type=str, required=True, help='Path to folder containing prediction audio files')
parser.add_argument('-tgt', dest='tgt', type=str, required=True, help='Path to folder containing target audio files')
args = parser.parse_args()

# Parameters
SAMPLE_RATE = 22050
FRAME_SIZE = 2048
HOP_SIZE = 512

def read_directory(directory):
    '''
    Read directory of audio files

    Inputs
        directory - Path to directory of audio files
    Outputs
        waveforms - Dictionary of (file number, waveform) items
    '''

    # Go through all audio files in directory
    waveforms = dict()
    for fname in os.listdir(directory):

        # Skip non audio files in case there are any
        if '.wav' not in fname:
            continue

        # Get the file paths for the piano sample and corresponding guitar sample
        fpath = os.path.join(directory, fname)
        signal, sr = librosa.load(fpath, sr=SAMPLE_RATE)
        print('Signal shape:', signal.shape)

        # Store file name and signal
        waveform_num = fname.split('_')[-1]
        waveforms[waveform_num] = signal

        '''
        # Get spectrogram using STFT (freq bins x num frames)
        spectrum = librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        spectrogram = np.abs(spectrum) ** 2
        log_spectrogram = librosa.power_to_db(spectrogram)

        # Plot spectrogram
        print('Showing:',prediction_fpath)
        plt.figure(figsize=(25,10))
        librosa.display.specshow(spectrogram, sr=sr, hop_length=HOP_SIZE,x_axis='time',y_axis='linear')
        librosa.display.specshow(log_spectrogram, sr=sr, hop_length=HOP_SIZE,x_axis='time',y_axis='log')
        plt.colorbar(format='%+2.f"')
        plt.show()
        '''
    
    return waveforms

# Load all prediction audio files
predictions = read_directory(args.pred)

# Load all target audio files
targets = read_directory(args.target)

# Calculate loss (MAE between predictions/targets) 
# and other statistics/visualizations
total_mae = 0
num_samples = 0
for file_num, waveform in predictions.items():

    # Get corresponding prediction/target waveforms
    pred_waveform = waveform
    target_waveform = targets[file_num]

    # Calculate statistics
    mae = np.sum(np.absolute((pred_waveform - target_waveform)))

    # Update overall statistics
    total_mae += mae
    num_samples += 1
