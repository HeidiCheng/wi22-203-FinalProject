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
class A_Matrices(Dataset):

    def __init__(self, directory, set_type, device):

        """
        directory - Directory that contains piano1/guitar1 folder of samples
                    and train, val, test split text files
                    (Assumes subdirectories '2', '3', '4', '5', '6')
        set_type - indicates if train, valid, or test
        device - device (cuda if GPU, cpu if CPU)
        """

        # Read set file (train/val/test) containing the list of samples to use
        A_list_dir = os.path.join(directory, 'A_temp_results')
        A_mats = []
        for fname in os.listdir(A_list_dir):
            with open(os.path.join(A_list_dir, fname), 'r') as f:
                lines = f.readlines()
                num_mats = int(len(lines)/9000)
                for m in range(num_mats):
                    A = [float(l) for l in lines[9000*m:9000*(m+1)]]
                    A = np.array(A)
                    A_mats.append(A)
        print('Number of matrices:', len(A_mats))

        self.matrices = A_mats
        self.device = device


    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        return self.matrices[idx]


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

        self.samples = []

        print("============= Load " + set_type + " data ==============")
        for i in trange(len(sample_fpaths)):

            # Get the file paths for the piano sample and corresponding guitar sample
            piano_fpath = os.path.join(directory, os.path.join('piano1_chords', sample_fpaths[i]))
            guitar_fpath = os.path.join(directory, os.path.join('guitar1_chords', sample_fpaths[i]))
             
            # Load audio files at 3000 sample rate
            piano_wav = torch.tensor(librosa.load(piano_fpath, sr=3000)[0])
            guitar_wav = torch.tensor(librosa.load(guitar_fpath, sr=3000)[0])

            #if i == 0:
            #    wavf.write('pianotest.wav', 3000, piano_wav.cpu().numpy())
            #    wavf.write('guitartest.wav', 3000, guitar_wav.cpu().numpy())

            # Store sample
            self.samples.append((piano_wav, guitar_wav))

            #if len(self.samples) == 20:
            #    break

        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        piano_wav, guitar_wav = self.samples[idx]

        return piano_wav.to(self.device), \
               guitar_wav.to(self.device)
