import torch
import lightning.pytorch as pl
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, random_split

class data_module(pl.LightningDataModule):

    def __init__(self, batch_size, seed, folder="data/"):
        super(data_module,self).__init__()
        self.folder=folder
        self.batch_size=batch_size
        self.seed=seed

    def setup(self, stage=None):

        print(os.path.join(self.folder,"X.csv"))

        self.X=np.loadtxt(os.path.join(self.folder,"X.csv"), dtype=np.float32, delimiter=',')
        self.X=np.delete(self.X,0,1) # Deleting the first column as it was saved as a pd DataFrame
        self.y=np.loadtxt(os.path.join(self.folder,"y.csv"), dtype=np.float32, delimiter=',')
        self.dataset=TensorDataset(torch.from_numpy(self.X),torch.from_numpy(self.y))

        self.train_dataset,self.val_dataset,self.test_dataset=random_split(self.dataset,[0.8,0.1,0.1],generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):        
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)    