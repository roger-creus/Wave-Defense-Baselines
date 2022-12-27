import os
import cv2
import torch
import numpy as np
import matplotlib.pylab as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from random import shuffle
from IPython import embed
import torch.nn.functional as F

from pathlib import Path

class CustomWaveDefenseData(Dataset):
    def __init__(self, path, traj_list, delay=False):
        
        self.path = Path(path) 
        self.traj_list = traj_list
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.customLoad()

    def getImage(self, idx):
        query = torch.FloatTensor(self.data[idx])
        return query

    def customLoad(self):
        print("Loading data...")
        data = []

        for i, traj in enumerate(self.traj_list):
            print(f"\tTraj: {i}", end ='\r')
            obs = np.load(str(self.path) + "/" + traj, allow_pickle=True)
            data.append(obs.flatten())
    
        data = np.concatenate(np.array(data, dtype='object'))
        data = data.reshape(-1, 84, 84, 3)
        self.data = data

        print("Loaded data of shape: " + str(data.shape))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # Get query obs
        x = self.getImage(index)

        # for RGB images only
        x = torch.div(x.unsqueeze(0).permute(0,3,1,2), 255)
        return x

