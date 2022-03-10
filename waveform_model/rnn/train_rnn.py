# Code for training timbre transfer using spectrogram as input
# with *insert architecture/model* name
# Run using the following:
#   python train_wav.py -data <path> -l <trained_model>
#
# python train_wav.py -data "/Users/heidicheng/Desktop/heidi/Convex_Optimization/project/data"

import torch
import torch.nn as nn
import torchaudio
import argparse
import model_rnn as model
import numpy as np
import random
from tqdm import tqdm

from torch.utils.data import DataLoader
from data_rnn import PianoToGuitar

# Set seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Hyperparams (define hyperparams)
epochs = 100
learning_rate = 1e-4
batch_size = 10
sample_rate = 44100
#sample_rate = 22050
input_size = 441
hidden_size = 256
embedded_size = 64
hyperparam_list = ['epochs', 'learning_rate', 'batch_size', 'sample_rate']
hyperparams = {name:eval(name) for name in hyperparam_list}

# Setup GPU stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using',device)

# Parse args
parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-data', dest='data', type=str, required=True, help='Path to the dataset')
parser.add_argument('-l', dest='load', type=str, required=False, help='Path to trained model')
parser.add_argument('-epochs', dest='epochs', type=int, default=100, help='Training epochs')
args = parser.parse_args()

# Load datasets
dataset_train = PianoToGuitar(args.data, 'train', device)
dataset_valid= PianoToGuitar(args.data, 'valid', device)

# Log information about dataset and training
print("========= List of Hyperparams ========")
for k,v in hyperparams.items():
    print(k,':',v)
print('Num training samples:', len(dataset_train))
print('Num validation samples:', len(dataset_valid))

# Create dataloaders
dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
dataloader_valid = DataLoader(dataset_valid, shuffle=True, batch_size=batch_size)

# Create model
model = model.WavRNN(input_size, hidden_size, embedded_size)
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
    root_model_path = '/content/drive/MyDrive/Colab Notebooks/trained_model/latest_model_' + str(model_num) + '.pt'
    model_dict = model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state_dict, root_model_path)

    print('Saved model')

# Define loss functions used
mae_loss = nn.L1Loss()

# Go through training data
print("=========== Start training ==========")
for epoch in range(args.epochs):

    # Reset statistics
    train_loss = 0
    num_corr = 0
    attr_acc = 0
    total = 0
    print('Epoch:', epoch)

    # Training loop
    model.train()
    for samples_piano, samples_guitar in tqdm(dataloader_train):

        # Reset gradient
        optimizer.zero_grad()

        # Forward pass
        piano_transformed = model(samples_piano)

        # Pad zeros to tensor to match shape of target since
        # dimensionality (height/width) was messed up slightly with convs
        zeros = torch.zeros((batch_size, 1, samples_guitar.shape[2] - piano_transformed.shape[2]), device=device)
        piano_transformed_shaped = torch.cat((piano_transformed, zeros), dim=2)

        # Calculate loss
        loss = mae_loss(piano_transformed_shaped, samples_guitar)   
        
        # Calculate accuracy
        
        # Backward pass (update)
        loss.backward()
        optimizer.step()

        # Update statistics
        train_loss += loss.item()
        total += batch_size #batch size

    # Show current statistics on training
    print('Train Loss:',train_loss / (len(dataloader_train) / batch_size) )

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
            zeros = torch.zeros((batch_size, 1, samples_guitar.shape[2] - piano_transformed.shape[2]), device=device)
            piano_transformed_shaped = torch.cat((piano_transformed, zeros), dim=2)

            # Calculate loss
            loss = mae_loss(piano_transformed_shaped, samples_guitar)   

            # Calculate accuracy

            # Update statistics
            valid_loss += loss.item()
            total += batch_size

    # Write a sample output
    piano_transformed_shaped = piano_transformed_shaped.cpu()
    samples_piano = samples_piano.cpu()
    samples_guitar = samples_guitar.cpu()
    torchaudio.save(args.data + '/output/input.wav', samples_piano[0], sample_rate, format='wav')
    torchaudio.save(args.data + '/output/pred.wav', piano_transformed_shaped[0], sample_rate, format='wav')
    torchaudio.save(args.data + '/output/tgt.wav', samples_guitar[0], sample_rate, format='wav')



    # Show statistics on test set
    print('Valid Loss:',valid_loss / (len(dataloader_valid) / batch_size))

    #if not args.test:
    if (epoch+1) % 10 == 0:
        save_model()
        model_num += 1