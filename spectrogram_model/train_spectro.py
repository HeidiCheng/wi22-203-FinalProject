# Code for training timbre transfer using spectrogram as input
# with *insert architecture/model* name
# Run using the following:
#   python train_spectro.py -data <path> -l <trained_model>
#
# python train_spectro.py -data "F:\Datasets\TimbreTransfer"

import torch
import torch.nn as nn
import argparse
import model
import numpy as np
import random
import librosa
import soundfile as sf

from torch.utils.data import DataLoader
from data import PianoGuitar_SS

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Hyperparams (define hyperparams)
epochs = 100
learning_rate = 1e-4
batch_size = 10
sample_rate = 22050
hyperparam_list = ['epochs', 'learning_rate', 'batch_size', 'sample_rate']
hyperparams = {name:eval(name) for name in hyperparam_list}

# Setup GPU stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using',device)

# Parse args
parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-data', dest='data', type=str, required=True, help='Path to the dataset')
parser.add_argument('-l', dest='load', type=str, required=False, help='Path to trained model')
args = parser.parse_args()

# Load datasets
dataset_train = PianoGuitar_SS(args.data, 'train', device)
dataset_valid= PianoGuitar_SS(args.data, 'valid', device)

# Log information about dataset and training
print('List of Hyperparams:')
for k,v in hyperparams.items():
    print(k,':',v)
print('Num training samples:', len(dataset_train))
print('Num validation samples:', len(dataset_valid))

# Create dataloaders
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
dataloader_valid = DataLoader(dataset_valid, shuffle=True, batch_size=batch_size)

# Create model
model = model.SpectroNet()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model_num = 1

# Load previous model if flag used
if args.load:
    state_dict = torch.load(args.load)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    print('Model loaded!', args.load)
    try:
        model_num = int(args.load.split('_')[-1].split('.')[0]) + 1
    except ValueError:
        pass

# Function to save model
def save_model():

    # Save model
    root_model_path = 'trained_models/latest_model_' + str(model_num) + '.pt'
    model_dict = model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state_dict, root_model_path)

    print('Saved model')

# Define loss functions used
#mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()

# Go through training data
for epoch in range(epochs):

    # Reset statistics
    train_loss = 0
    num_corr = 0
    attr_acc = 0
    total = 0
    print('Epoch:', epoch)

    # Training loop
    model.train()
    for samples_piano, samples_guitar in dataloader_train:

        #print('Training shapes')
        #print(samples_piano.shape, samples_guitar.shape)

        # Reset gradient
        optimizer.zero_grad()

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
        
        # Backward pass (update)
        loss.backward()
        optimizer.step()

        # Update statistics
        train_loss += loss.item()
        total += 10 #batch size

    # Show current statistics on training
    print('Train Loss:',train_loss / (len(dataloader_train) / batch_size) )
    #print('Train Class Accuracy:',(num_corr / total).item())
    #print('Train Attribute Accuracy:',(attr_acc / total))

    # Reset statistics
    valid_loss = 0
    num_corr = 0
    attr_acc = 0
    total = 0

    # Validation loop
    model.eval()
    for samples_piano, samples_guitar in dataloader_valid:

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
            valid_loss += loss.item()
            total += 10 #batch size

    # Write a sample output
    piano_transformed_shaped = piano_transformed_shaped.cpu().numpy()
    samples_piano = samples_piano.cpu().numpy()
    samples_guitar = samples_guitar.cpu().numpy()
    audio_reconstructed_input =librosa.griffinlim(samples_piano[0,0])
    audio_reconstructed_model = librosa.griffinlim(piano_transformed_shaped[0,0])
    audio_reconstructed_tgt = librosa.griffinlim(samples_guitar[0,0])
    sf.write('input.wav', audio_reconstructed_input, sample_rate, 'PCM_24')
    sf.write('pred.wav', audio_reconstructed_model, sample_rate, 'PCM_24')
    sf.write('tgt.wav', audio_reconstructed_tgt, sample_rate, 'PCM_24')

    # Show statistics on test set
    print('Valid Loss:',valid_loss / (len(dataloader_valid) / batch_size))
    #print('Valid Class Accuracy:',(num_corr / total).item())
    #print('Valid Attribute Accuracy:',(attr_acc / total))

    #if not args.test:
    if (epoch+1) % 10 == 0:
        save_model()
        model_num += 1