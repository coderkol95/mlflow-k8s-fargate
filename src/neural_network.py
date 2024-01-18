import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List

class NN(pl.LightningModule):

    def __init__(
            self, 
            dropout:float,
            output_dims: List[int],
            lr:float) -> None:

        super().__init__()
        self.lr=lr
        self.loss=F.mse_loss
        self.output_dims=output_dims
        self.dropout=dropout

        layers: List[nn.Module] = []

        input_dim: int = 5
        for output_dim in self.output_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 1))

        self.layers = nn.Sequential(*layers)

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=self.lr)

    def forward(self, data):
        # If logits were returned, you would have returned the F.softmax etc. 
        return self.layers(data)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch 
        preds = self.forward(x)
        loss = self.loss(preds.flatten(), y) 
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, valid_batch, batch_idx): 
        x, y = valid_batch 
        preds = self.forward(x) 
        loss = self.loss(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss