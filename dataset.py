"""Datasets definitions for tactile data.
dataset should return an output of shape (data, target, label).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
from utils.utils import letters

class TactileDataset(Dataset):
    def __init__(
        self,
        path,
        fold,
        trial_number,
        output_size,
        split_name='train'
    ):
        self.output_size=output_size
        
        assert split_name in ['train', 'validation', 'test'], print(f'split_name is incorrect {split_name}')

        # load data
        splits = json.load(open(Path(path) / 'splits.txt', 'r'))
        self.samples = splits[f'trial{trial_number}']
        self.info = np.loadtxt(Path(path) / 'info.txt', delimiter=',', dtype=str) # letter sample_id data_ind
        data = torch.FloatTensor( np.load(Path(path) / "data.npy") )
        

        # calculate size
        self.size = 0
        print(split_name)
        for letter in letters:
            if split_name in ['train', 'validation']:
                self.size += len(self.samples[letter][split_name][str(fold)])
            else:
                self.size += len(self.samples[letter][split_name])
        
        # bring to SLAYER format
        self.data = data.reshape(data.shape[0], -1, 1, 1, data.shape[-1])


    def __getitem__(self, index):
        input_index = int(self.info[index, 2])
        class_label = int(self.info[letters[index], 1])
        target_class = torch.zeros((self.output_size, 1, 1, 1))
        target_class[class_label, ...] = 1

        return self.data[input_index], target_class, class_label

    def __len__(self):
        return self.size