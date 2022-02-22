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

from torch.utils.data import Dataset

# Dataset class for loading PianoGuitar_SingleSound dataset
# PianoGuitar indicates only two instruments are piano and guitar
# SingleSound indicates only one piano sound and one guitar sound was used
class PianoGuitar_SS(Dataset):

    def __init__(self, directory, set_type, device):

        """
        directory - Directory that contains piano1/guitar1 folder of samples
                    and train, val, test split text files
                    (Assumes subdirectories '2', '3', '4', '5', '6')
        set_type - indicates if train, valid, or test
        device - device (cuda if GPU, cpu if CPU)
        """

        # Read set file (train/val/test) containing the list of samples to use
        image_list_fname = os.path.join(directory, set_type + '.txt')
        sample_fpaths = []
        with open(image_list_fname, 'r') as f:
            for l in f.readlines():
                l = l.strip().split(',')
                dir_name = l[0]
                # f_name = l[1] + '.wav'    # Use for wav files
                f_name = l[1] + '.npy'      # Use for spectrograms
                sample_fpaths.append(os.path.join(dir_name, f_name))
        
        # Parameters for STFT
        FRAME_SIZE = 2048
        HOP_SIZE = 512
        SAMPLE_RATE = 22050

        # Load all audio samples to be used in this set (train/val/test)
        self.samples = []
        for sample_fpath in sample_fpaths:

            '''
            # Load audio files and then use STFT for getting spectrogram

            # Get the file paths for the piano sample and corresponding guitar sample
            piano_fpath = os.path.join(directory, os.path.join('piano1_chords', sample_fpath))
            guitar_fpath = os.path.join(directory, os.path.join('guitar1_chords', sample_fpath))
            
            # Load audio files
            piano_signal, sr = librosa.load(piano_fpath, sr=SAMPLE_RATE)
            guitar_signal, _ = librosa.load(guitar_fpath, sr=SAMPLE_RATE)
 
            # Get spectrogram using STFT (freq bins x num frames)
            piano_s = librosa.stft(piano_signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
            piano_spectrogram = np.abs(piano_s) # ** 2
            guitar_s = librosa.stft(guitar_signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
            guitar_spectrogram = np.abs(guitar_s) # ** 2
            '''

            # Load pre-saved spectrograms directly instead of audio files

            # Get the file paths for the piano sample and corresponding guitar sample
            piano_fpath = os.path.join(directory, os.path.join('piano1_spectro', sample_fpath))
            guitar_fpath = os.path.join(directory, os.path.join('guitar1_spectro', sample_fpath))
            
            # Load spectrograms 
            with open(piano_fpath, 'rb') as f:
                piano_spectrogram = np.load(f) # ** 2
            with open(guitar_fpath, 'rb') as f:
                guitar_spectrogram = np.load(f) # ** 2

            #print(guitar_spectrogram.shape)

            # Save spectrograms
            self.samples.append((piano_spectrogram, guitar_spectrogram))

            # Track data loading progress
            if len(self.samples) % 1000 == 0:
                print(len(self.samples))

            # Audio reconstruction from spectrogram
            # audio_reconstructed = librosa.griffinlim(spectrogram)

            #log_spectrogram = librosa.power_to_db(spectrogram)
            #audio_reconstructed = librosa.griffinlim(spectrogram)
            #print('Orig:', signal.shape, 'Reconst:', audio_reconstructed.shape)
            #sf.write('test.wav', audio_reconstructed, SAMPLE_RATE, 'PCM_24')

            # Plot spectrogram
            #print('Showing:',piano_fpath)
            #plt.figure(figsize=(25,10))
            #librosa.display.specshow(spectrogram, sr=sr, hop_length=HOP_SIZE,x_axis='time',y_axis='linear')
            #librosa.display.specshow(log_spectrogram, sr=sr, hop_length=HOP_SIZE,x_axis='time',y_axis='log')
            #plt.colorbar(format='%+2.f"')
            #plt.show()
            
            '''
            fs_rate, signal = wavfile.read(piano_fpath)
            print(piano_fpath)
            print('Sampling rate:', fs_rate)
            print('Signal shape:', signal.shape)
            l_audio = len(signal.shape)
            print ("Channels", l_audio)
            if l_audio == 2:   # Two channel, average them
                signal = signal.sum(axis=1) / 2
                print('Signal shape averaged:', signal.shape)
                print('Range of signal:', np.max(signal), np.min(signal), np.mean(signal))
            N = signal.shape[0]
            print ("Number of samplings", N)
            secs = N / float(fs_rate)
            print ("Sample Length (seconds)", secs)
            Ts = 1.0/fs_rate # sampling interval in time
            print ("Timestep between samples Ts", Ts)
            t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
            FFT = abs(fft(signal))
            print('FFT Shape:', FFT.shape)
            FFT_side = FFT[range(N//2)] # one side FFT range
            freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
            fft_freqs = np.array(freqs)
            freqs_side = freqs[range(N//2)] # one side frequency range
            fft_freqs_side = np.array(freqs_side)
            plt.subplot(311)
            p1 = plt.plot(t, signal, "g") # plotting the signal
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.subplot(312)
            p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Count dbl-sided')
            plt.subplot(313)
            p3 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Count single-sided')
            plt.show()
            #with open(piano_fpath, 'r') as f:
            #    print(piano_fpath)
            '''

        
        #print('Number of {} samples: {}'.format(set_type, len(self.samples)))

        self.device = device


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        piano_spectro, guitar_spectro = self.samples[idx]

        # Define image normalizaiton/preprocessing
        #preprocess = T.Compose([
            #T.Resize((224, 224)),
            #T.ToTensor(),
            #T.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225]
            #)
        #])

        return torch.tensor(piano_spectro, device=self.device), \
               torch.tensor(guitar_spectro, device=self.device)