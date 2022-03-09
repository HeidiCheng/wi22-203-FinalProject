# Code for training timbre transfer using spectrogram as input
# with *insert architecture/model* name
# Run using the following:
#   python train_spectro.py -data <path> -l <trained_model>
#
# python train_spectro.py -data "F:\Datasets\TimbreTransfer"

import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import librosa
import soundfile as sf

from torch.utils.data import DataLoader
from data import PianoToGuitar

import scipy.io.wavfile as wavf

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Setup GPU stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using',device)

# Parse args
parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-data', dest='data', type=str, required=True, help='Path to the dataset')
parser.add_argument('-l', dest='load', type=str, required=False, help='Path to trained model')
args = parser.parse_args()

# Load datasets
dataset_train = PianoToGuitar(args.data, 'train', device)
dataset_test = PianoToGuitar(args.data, 'test', device)

# Get training piano/guitar samples
piano_matrix = dataset_train.piano_samples
guitar_matrix = dataset_train.guitar_samples

# Minimize_A ||AX - B|| where X is piano sample, B is guitar sample
# AX = B
# A = B @ pinv(X)
X = piano_matrix
B = guitar_matrix
A = B @ (np.linalg.inv(X.T @ X) @ X.T) #np.linalg.pinv(X)

# Testing stuff
#A = piano_matrix.T
#B = guitar_matrix.T
#X = np.linalg.inv(A.T @ A) @ A.T @ B
#print(X.shape)

# Get test piano/guitar samples
test_piano_matrix = dataset_test.piano_samples
test_guitar_matrix = dataset_test.guitar_samples

for i in range(5):
    guitar_reconstructed = A @ test_piano_matrix[:,i]
    #guitar_reconstructed = test_piano_matrix[:,i].T @ X
    sf.write('input{}.wav'.format(str(i)), test_piano_matrix[:,i], 3000, 'PCM_24')
    sf.write('pred{}.wav'.format(str(i)), guitar_reconstructed, 3000, 'PCM_24')
    sf.write('tgt{}.wav'.format(str(i)), test_guitar_matrix[:,i], 3000, 'PCM_24')

