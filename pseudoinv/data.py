# Code for loading datasets for timbre transfer dataset 
# (Piano) <-> (Guitar) for CSE 203B project

import torch
import os
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image

# Audio processing stuff
import librosa
import librosa.display
import scipy
from scipy.io import wavfile # get the api
from scipy.fftpack import fft
#import scipy.fftpack
from matplotlib import pyplot as plt
import soundfile as sf

import torchaudio
import torchaudio.transforms as T
from tqdm import trange
import scipy.io.wavfile as wavf

from torch.utils.data import Dataset

# Dataset class for loading PianoGuitar_SingleSound dataset
# PianoGuitar indicates only two instruments are piano and guitar
# SingleSound indicates only one piano sound and one guitar sound was used
class PianoToGuitar(Dataset):

    def __init__(self, directory, set_type, device):
        list_fname = os.path.join(directory, set_type + '_set.txt')
        #list_fname = os.path.join(directory, 'valid_set.txt')
        sample_fpaths = []

        with open(list_fname, 'r') as f:
            for l in f.readlines():
                l = l.strip().split(',')
                dir_name = l[0]
                f_name = l[1] + '.wav' 
                sample_fpaths.append(os.path.join(dir_name, f_name))

        self.piano_samples = []
        self.guitar_samples = []
        print("============= Load " + set_type + " data ==============")
        for i in trange(len(sample_fpaths)):

            # Get the file paths for the piano sample and corresponding guitar sample
            piano_fpath = os.path.join(directory, os.path.join('piano1_chords', sample_fpaths[i]))
            guitar_fpath = os.path.join(directory, os.path.join('guitar1_chords', sample_fpaths[i]))
             
            # Load audio files at 3000 sample rate
            piano_wav = librosa.load(piano_fpath, sr=3000)[0]
            guitar_wav = librosa.load(guitar_fpath, sr=3000)[0]

            # Store sample
            self.piano_samples.append(piano_wav)
            self.guitar_samples.append(guitar_wav)

            if len(self.piano_samples) == 100:
                break

        # Convert to numpy matrix
        self.piano_samples = np.array(self.piano_samples).T
        self.guitar_samples = np.array(self.guitar_samples).T

        print('Piano shape:', self.piano_samples.shape)
        print('Guitar shape:', self.guitar_samples.shape)

        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        piano_wav, guitar_wav = self.samples[idx]

        return piano_wav.to(self.device), \
               guitar_wav.to(self.device)
