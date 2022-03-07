import os
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import Dataset
from tqdm import trange

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

            # Load audio files
            # Get the file paths for the piano sample and corresponding guitar sample
            piano_fpath = os.path.join(directory, os.path.join('piano1_chords', sample_fpaths[i]))
            guitar_fpath = os.path.join(directory, os.path.join('guitar1_chords', sample_fpaths[i]))
             
            piano_wav = torchaudio.load(piano_fpath)
            guitar_wav = torchaudio.load(guitar_fpath)

            resampler = T.Resample(piano_wav[1], 22016, dtype=piano_wav[0].dtype)

            piano_wav_downsampled = resampler(piano_wav[0])
            guitar_wav_downsampled = resampler(guitar_wav[0])

            mono_piano_wav = torch.mean(piano_wav_downsampled, dim=0).unsqueeze(0)
            mono_guitar_wav = torch.mean(guitar_wav_downsampled, dim=0).unsqueeze(0)

            self.samples.append((mono_piano_wav, mono_guitar_wav))

        self.device = device


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        piano_wav, guitar_wav = self.samples[idx]

        return piano_wav.to(self.device), \
               guitar_wav.to(self.device)
