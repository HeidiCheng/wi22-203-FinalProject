# Code for training timbre transfer using spectrogram as input
# with *insert architecture/model* name
# Run using the following:
#   python train_spectro.py -data <path> -l <trained_model>

import torch
import torch.nn as nn
import argparse

from torch.utils.data import DataLoader
from data import PianoGuitar_SS

# Hyperparams (define hyperparams)
epochs = 20
hyperparam_list = []
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
dataset_train = PianoGuitar_SS(args.data, 'train')
dataset_valid= PianoGuitar_SS(args.data, 'valid')

# Log information about dataset and training
print('List of Hyperparams:')
for k,v in hyperparams.items():
    print(k,':',v)
print('Num training samples:', len(dataset_train))
print('Num validation samples:', len(dataset_valid))

# Create dataloaders
dataloader_train = DataLoader(dataset_train, shuffle=True)
dataloader_valid = DataLoader(dataset_valid, shuffle=True)

# Create model
model = 1 # Model definition
model.to(device)
optimizer = 1 # Optimizer definition
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
    root_model_path = 'models/latest_model_' + str(model_num) + '.pt'
    model_dict = model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': optimizer.state_dict()}
    torch.save(state_dict, root_model_path)

    print('Saved model')

# Go through training data
for epoch in range(epochs):

    # Reset statistics
    train_loss = 0
    num_corr = 0
    attr_acc = 0
    total = 0
    print('Epoch:', epoch)

    # Training loop
    if not args.test:
        model.train()
        for samples_piano, samples_guitar in dataloader_train:
            
            samples_piano, samples_guitar = samples_piano.to(device), samples_guitar.to(device)

            # Reset gradient
            optimizer.zero_grad()

            # Forward pass
            _ = model(images)

            # Calculate loss
            loss = 1
            
            # Calculate accuracy

            # Backward pass (update)
            loss.backward()
            optimizer.step()

            # Update statistics
            train_loss += loss.item()
            total += 16 #batch size

        # Show current statistics on training
        print('Train Loss:',train_loss / len(dataloader_train))
        print('Train Class Accuracy:',(num_corr / total).item())
        print('Train Attribute Accuracy:',(attr_acc / total))

    # Reset statistics
    test_loss = 0
    num_corr = 0
    attr_acc = 0
    total = 0

    # Validation/test (if eval mode) loop
    model.eval()
    for samples_piano, samples_guitar in dataloader_valid:

        with torch.no_grad():

            samples_piano, samples_guitar = samples_piano.to(device), samples_guitar.to(device)

            # Forward pass
            _ = model(samples_piano)

            # Calculate loss
            loss = 1

            # Calculate accuracy

            # Update statistics
            test_loss += loss.item()
            total += 16 # batch size

    # Show statistics on test set
    print('Test Loss:',test_loss / len(dataloader_valid))
    print('Test Class Accuracy:',(num_corr / total).item())
    print('Test Attribute Accuracy:',(attr_acc / total))

    if not args.test:
        save_model()
        model_num += 1