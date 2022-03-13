# Code for evaluating piano to guitar sound transfer
# Name the files as below so they correspond exactly to test folder
# 2_0102.wav, 3_042.wav, etc.
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

        # Store file name and signal
        waveforms[fname] = signal

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
targets = read_directory(args.tgt)

# Calculate loss (MAE between predictions/targets) 
# and other statistics/visualizations
total_mae = 0
total_mse = 0
num_samples = 0
chord_num_mae = [0 for i in range(5)]
chord_num_mse = [0 for i in range(5)]
chord_num_total = [0 for i in range(5)]
for file_num, waveform in predictions.items():

    # Get corresponding prediction/target waveforms
    pred_waveform = waveform
    target_waveform = targets[file_num]

    # Calculate statistics
    mae = np.sum(np.absolute((pred_waveform - target_waveform[:len(pred_waveform)])))
    mse = np.sum(np.square((pred_waveform - target_waveform[:len(pred_waveform)])))

    # Update overall statistics
    total_mae += mae
    total_mse += mse
    num_samples += 1
    chord_num_mae[int(file_num[0])-2] += mae
    chord_num_mse[int(file_num[0])-2] += mse
    chord_num_total[int(file_num[0])-2] += 1

print('MAE:', total_mae / num_samples)
print('MSE:', total_mse / num_samples)

print('\nError by Chord Size')
for i in range(5):
    print('Chord size {} : MSE = {}    MAE = {}'.format(
        i+2, chord_num_mae[i] / chord_num_total[i], chord_num_mse[i] / chord_num_total[i]))