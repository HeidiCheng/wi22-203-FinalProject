import os
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import Dataset

class PianoToGuitar(Dataset):

    def __init__(self, directory, set_type, device):
      self.set_type = set_type
      list_fname = os.path.join(directory, set_type + '_set.txt')
      #list_fname = os.path.join(directory, 'valid_set.txt')
      sample_fpaths = []

      with open(list_fname, 'r') as f:
          for l in f.readlines():
              l = l.strip().split(',')
              dir_name = l[0]
              f_name = l[1] + '.wav' 
              sample_fpaths.append((os.path.join(dir_name, f_name), dir_name))

      self.samples = []
      for sample_fpath, dir_name in sample_fpaths:

          # Load audio files
          # Get the file paths for the piano sample and corresponding guitar sample
          piano_fpath = os.path.join(directory, os.path.join("chords", os.path.join('piano1_chords', sample_fpath)))
          guitar_fpath = os.path.join(directory, os.path.join("chords", os.path.join('guitar1_chords', sample_fpath)))
            
          piano_wav = torchaudio.load(piano_fpath)
          guitar_wav = torchaudio.load(guitar_fpath)
          resampler = T.Resample(piano_wav[1], 5000, dtype=piano_wav[0].dtype)
          piano_wav_downsampled = resampler(piano_wav[0])
          guitar_wav_downsampled = resampler(guitar_wav[0])

          fname = dir_name + "_" + sample_fpath.split("/")[-1].split(".")[0]
          self.samples.append((piano_wav_downsampled, guitar_wav_downsampled, fname))

          # Track data loading progress
          if len(self.samples) % 1000 == 0:
              print(len(self.samples))

      self.device = device


    def __len__(self):
      return len(self.samples)

    def __getitem__(self, idx):

      piano_wav, guitar_wav, fname = self.samples[idx]

      mono_piano_wav = torch.mean(piano_wav, dim=0).unsqueeze(0)
      mono_guitar_wav = torch.mean(guitar_wav, dim=0).unsqueeze(0)

      if self.set_type == "test":
        return mono_piano_wav.to(self.device), \
                mono_guitar_wav.to(self.device), \
                fname
      
      return mono_piano_wav.to(self.device), \
              mono_guitar_wav.to(self.device)

