# Code for loading datasets for timbre transfer dataset 
# (Piano) <-> (Guitar) for CSE 203B project

import torch
import os
import random
import numpy as np
import torchvision.transforms as T
from PIL import Image

from torch.utils.data import Dataset

# Dataset class for loading PianoGuitar_SingleSound dataset
# PianoGuitar indicates only two instruments are piano and guitar
# SingleSound indicates only one piano sound and one guitar sound was used
class PianoGuitar_SS(Dataset):

    def __init__(self, directory, set_type):

        """
        directory - Directory that contains piano1/guitar1 folder of samples
                    and train, val, test split text files
                    (Assumes subdirectories '2', '3', '4', '5', '6')
        set_type - indicates if train, valid, or test
        """

        # Read set file (train/val/test) containing the list of samples to use
        image_list_fname = os.path.join(directory, set_type + '.txt')
        with open(image_list_fname, 'r') as f:
            for l in f.readlines():
                print(l)
        
        # Load all audio samples to be used in this set (train/val/test)
       


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img, label = self.samples[idx]

        # Define image normalizaiton/preprocessing
        #preprocess = T.Compose([
            #T.Resize((224, 224)),
            #T.ToTensor(),
            #T.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225]
            #)
        #])

        return torch.tensor(img), torch.tensor(label)

    def training_init(self, directory, set_type):
        '''
        Initialization for training, uses training/val split from
        PS files such as trainclasses1.txt, valclasses1.txt
        '''

        # Read PS train/val/test splits
        train_list_fname = os.path.join(directory, os.path.join('PS', 'trainclasses1.txt'))
        val_list_fname = os.path.join(directory, os.path.join('PS', 'valclasses1.txt'))
        test_list_fname = os.path.join(directory, os.path.join('PS', 'testclasses.txt'))
        with open(train_list_fname, 'r') as train_file, open(val_list_fname, 'r') as val_file,\
            open(test_list_fname, 'r') as test_file:
            self.train_classes = [a.strip() for a in train_file.readlines()]
            self.val_classes = [a.strip() for a in val_file.readlines()]
            self.test_classes = [a.strip() for a in test_file.readlines()]
        
        # Determine set classes based on param
        if set_type == 'train':
            self.set_classes = self.train_classes
        elif set_type == 'valid':
            self.set_classes = self.val_classes
        else:
            self.set_classes = self.test_classes

        # Read images and create dataset
        self.samples = []
        for i, (img_path, label) in enumerate(zip(self.img_list_fpaths, self.img_list_labels)):

            # Skip classes not in the class set (train/test)
            if self.classes[label] not in self.set_classes:
                continue

            # Open image and append sample pair (img, label)
            fpath = os.path.join(directory, img_path)
            #img = Image.open(fpath).convert('RGB')
            img = self.image_feature_mapping[i] # Pretrained image feats (2048d)
            sample = (img, label)
            self.samples.append(sample)
            #if len(self.samples) % 1000 == 0:
            #    print(len(self.samples))

    def eval_init(self, directory, set_type):
        '''
        Initialization for evaluation, uses training/test split from
        PS files such as train_seen_loc.csv, test_seen_loc.csv, and
        test_unseen_loc.csv
        '''

        # For exact PS finally used after tuning on train/val
        train_seen_idxs = os.path.join(directory, os.path.join('PS', 'train_seen_loc.csv'))
        test_seen_idxs = os.path.join(directory, os.path.join('PS', 'test_seen_loc.csv'))
        test_unseen_idxs = os.path.join(directory, os.path.join('PS', 'test_unseen_loc.csv'))
        train_seen = []
        test_seen = []
        test_unseen = []
        with open(train_seen_idxs, 'r') as a, open(test_seen_idxs, 'r') as b, open(test_unseen_idxs, 'r') as c:
            for s in a.readlines():     # Train seen
                idx = int(s.strip()) - 1   # Matlab 1 indexing
                train_seen.append(idx)
            for s in b.readlines():     # Test seen
                idx = int(s.strip()) - 1   # Matlab 1 indexing
                test_seen.append(idx)
            for s in c.readlines():     # Test unseen
                idx = int(s.strip()) - 1   # Matlab 1 indexing
                test_unseen.append(idx)

        # Determine set classes based on param
        if set_type == 'train':
            self.set_classes = train_seen
        else:
            self.set_classes = test_seen + test_unseen

        # Store training/validation classes (val is test in this case of eval)
        self.train_classes = set()
        self.val_classes = set()
        for idx in train_seen:
            self.train_classes.add(self.classes[self.img_list_labels[idx]])
        for idx in test_seen + test_unseen:
            self.val_classes.add(self.classes[self.img_list_labels[idx]])

        # Read images and create dataset
        self.samples = []
        for i, (img_path, label) in enumerate(zip(self.img_list_fpaths, self.img_list_labels)):

            # Skip classes not in the class set (train/test)
            if i not in self.set_classes:
                continue

            # Open image and append sample pair (img, label)
            fpath = os.path.join(directory, img_path)
            #img = Image.open(fpath).convert('RGB')
            img = self.image_feature_mapping[i] # Pretrained image feats (2048d)
            sample = (img, label)
            self.samples.append(sample)
            #if len(self.samples) % 1000 == 0:
            #    print(len(self.samples))
