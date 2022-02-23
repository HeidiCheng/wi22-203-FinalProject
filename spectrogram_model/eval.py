# Code for evaluating timbre transfer using spectrogram as input
# with *insert architecture/model* name
# Run using the following:
#   python eval.py -data <path> -l <trained_model>
#
# python eval.py -data "F:\Datasets\TimbreTransfer"

import torch
import torch.nn as nn
import argparse
import model
import numpy as np
import random
import librosa
import soundfile as sf
import os

from torch.utils.data import DataLoader
from data import PianoGuitar_SS

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Hyperparams (define hyperparams)
batch_size = 10
sample_rate = 22050
hyperparam_list = ['batch_size', 'sample_rate']
hyperparams = {name:eval(name) for name in hyperparam_list}

# Setup GPU stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using',device)

# Parse args
parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-data', dest='data', type=str, required=True, help='Path to the dataset')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to trained model')
parser.add_argument('-out', dest='out', type=str, required=True, help='Path to output predictions to')
parser.add_argument('-f', dest='f', type=str, required=False, help='Path to single audio file (.wav) to predict for')
args = parser.parse_args()

# Load datasets
dataset_test = PianoGuitar_SS(args.data, 'test', device)

# Log information about dataset and training
print('List of Hyperparams:')
for k,v in hyperparams.items():
    print(k,':',v)

# Create dataloaders
dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

# Create model
model = model.SpectroNet()
model.to(device)

# Load trained model
state_dict = torch.load(args.model)
model.load_state_dict(state_dict['model'])
print('Model loaded!', args.model)

# Define loss functions used
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()

# Reset statistics
test_loss = 0
num_corr = 0
attr_acc = 0
total = 0

# Do evaluation
model.eval()
sample_num = 0
if not args.f:  # Not single file prediction

    for samples_piano, samples_guitar in dataloader_test:

        with torch.no_grad():

            # Forward pass
            piano_transformed = model(samples_piano)

            # Pad zeros to tensor to match shape of target since
            # dimensionality (height/width) was messed up slightly with convs
            zeros1 = torch.zeros((batch_size, 1, piano_transformed.shape[2],
                                samples_piano.shape[3] - piano_transformed.shape[3]), device=device)
            zeros2 = torch.zeros((batch_size, 1, samples_piano.shape[2] - piano_transformed.shape[2],
                                samples_piano.shape[3]), device=device)
            piano_transformed_shaped = torch.cat((piano_transformed, zeros1), dim=3)
            piano_transformed_shaped = torch.cat((piano_transformed_shaped, zeros2), dim=2)

            # Calculate loss
            #loss = mse_loss(piano_transformed_shaped, samples_guitar)
            loss = mae_loss(piano_transformed_shaped, samples_guitar)

            # Calculate accuracy

            # Update statistics
            test_loss += loss.item()
            total += 10 #batch size

            # Write outputs
            piano_transformed_shaped = piano_transformed_shaped.cpu().numpy()
            samples_piano = samples_piano.cpu().numpy()
            samples_guitar = samples_guitar.cpu().numpy()
            for k in range(batch_size):
                audio_reconstructed_input =librosa.griffinlim(samples_piano[k,0])
                audio_reconstructed_model = librosa.griffinlim(piano_transformed_shaped[k,0])
                audio_reconstructed_tgt = librosa.griffinlim(samples_guitar[k,0])
                sf.write(os.path.join(args.out, str(sample_num) + '_input' + '.wav'), audio_reconstructed_input, sample_rate, 'PCM_24')
                sf.write(os.path.join(args.out, str(sample_num) + '_pred' + '.wav'), audio_reconstructed_model, sample_rate, 'PCM_24')
                sf.write(os.path.join(args.out, str(sample_num) + '_tgt' + '.wav'), audio_reconstructed_tgt, sample_rate, 'PCM_24')
                sample_num += 1

    # Show statistics on test set
    print('Test Loss:',test_loss / (len(dataloader_test) / batch_size))
    #print('Valid Class Accuracy:',(num_corr / total).item())
    #print('Valid Attribute Accuracy:',(attr_acc / total))

else:
    # Single file prediction
    with torch.no_grad():

        # Load audio file
        piano_fpath = args.f
        piano_signal, sr = librosa.load(piano_fpath, sr=sample_rate)
 
        # Parameters for STFT
        FRAME_SIZE = 2048
        HOP_SIZE = 512
        SAMPLE_RATE = 22050

        # Get spectrogram using STFT (freq bins x num frames)
        piano_s = librosa.stft(piano_signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        samples_piano = torch.tensor(np.array([[np.abs(piano_s)]]), device=device)
        fname = args.f.split('\\')[-1]

        # Forward pass
        piano_transformed = model(samples_piano)

        # Pad zeros to tensor to match shape of target since
        # dimensionality (height/width) was messed up slightly with convs
        zeros1 = torch.zeros((1, 1, piano_transformed.shape[2],
                            samples_piano.shape[3] - piano_transformed.shape[3]), device=device)
        zeros2 = torch.zeros((1, 1, samples_piano.shape[2] - piano_transformed.shape[2],
                            samples_piano.shape[3]), device=device)
        piano_transformed_shaped = torch.cat((piano_transformed, zeros1), dim=3)
        piano_transformed_shaped = torch.cat((piano_transformed_shaped, zeros2), dim=2)

        # Write outputs
        piano_transformed_shaped = piano_transformed_shaped.cpu().numpy()
        samples_piano = samples_piano.cpu().numpy()
        audio_reconstructed_input =librosa.griffinlim(samples_piano[0,0])
        audio_reconstructed_model = librosa.griffinlim(piano_transformed_shaped[0,0])
        sf.write(os.path.join(args.out, 'input_' + fname), audio_reconstructed_input, sample_rate, 'PCM_24')
        sf.write(os.path.join(args.out, 'pred_' + fname), audio_reconstructed_model, sample_rate, 'PCM_24')