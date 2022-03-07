# Code for evaluating timbre transfer using waveform as input
# Run using the following:
#   python eval_wav_pitch.py -data <dataset path> -model <trained_model path> -out <output path> -f <testing audio file path>


import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import argparse
import model
import numpy as np
import soundfile as sf
import random
import os

from torch.utils.data import DataLoader
from data import PianoToGuitar

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
parser = argparse.ArgumentParser(description='Test model.')
parser.add_argument('-data', dest='data', type=str, required=True, help='Path to the dataset')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to trained model')
parser.add_argument('-out', dest='out', type=str, required=True, help='Path to output predictions to')
parser.add_argument('-f', dest='f', type=str, required=False, help='Path to single audio file (.wav) to predict for')
args = parser.parse_args()

# Load datasets
dataset_test = PianoToGuitar(args.data, 'test', device)

# Log information about dataset and training
print('List of Hyperparams:')
for k,v in hyperparams.items():
    print(k,':',v)

# Create dataloaders
dataloader_test = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

# Create model
model = model.WavNet()
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
            zeros = torch.zeros((batch_size, 1, samples_piano.shape[2] - piano_transformed.shape[2]), device=device)
            piano_transformed_shaped = torch.cat((piano_transformed, zeros), dim=2)

            # Add pitch detection to loss
            guitar_pitch = torchaudio.functional.detect_pitch_frequency(samples_guitar, sample_rate, 0.1)
            pred_pitch = torchaudio.functional.detect_pitch_frequency(piano_transformed_shaped, sample_rate, 0.1)

            # Calculate loss
            #loss = mse_loss(piano_transformed_shaped, samples_guitar)
            loss = mae_loss(piano_transformed_shaped, samples_guitar)
            loss += 0.1*mae_loss(torch.log(guitar_pitch), torch.log(pred_pitch))

            # Calculate accuracy

            # Update statistics
            test_loss += loss.item()
            total += 10 #batch size

            # Write outputs
            piano_transformed_shaped = piano_transformed_shaped.cpu()
            samples_piano = samples_piano.cpu()
            samples_guitar = samples_guitar.cpu()
            for k in range(batch_size):
                torchaudio.save(os.path.join(args.out, str(sample_num) + '_input' + '.wav'), samples_piano[k], sample_rate, format='wav')
                torchaudio.save(os.path.join(args.out, str(sample_num) + '_pred' + '.wav'), piano_transformed_shaped[k], sample_rate, format='wav')
                torchaudio.save(os.path.join(args.out, str(sample_num) + '_tgt' + '.wav'),samples_guitar[k], sample_rate, firmat='wav')
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
        piano_wav = torchaudio.load(piano_fpath)
 
        # Wav to tensor
        resampler = T.Resample(piano_wav[1], sample_rate, dtype=piano_wav[0].dtype)
        piano_wav_downsampled = resampler(piano_wav[0])

        mono_piano_wav = torch.mean(piano_wav_downsampled, dim=0).unsqueeze(0)
        mono_piano_wav = torch.reshape(mono_piano_wav, (1, 1, mono_piano_wav.shape[1])).to(device)
        fname = args.f.split('\\')[-1]

        # Forward pass
        piano_transformed = model(mono_piano_wav)

        # Write outputs
        piano_transformed = piano_transformed.cpu()
        mono_piano_wav = mono_piano_wav.cpu()
        torchaudio.save(os.path.join(args.out, 'input_' + fname), mono_piano_wav[0], sample_rate, format='wav')
        torchaudio.save(os.path.join(args.out, 'pred_' + fname), piano_transformed[0], sample_rate, format='wav')