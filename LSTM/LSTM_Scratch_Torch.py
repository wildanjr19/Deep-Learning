'''
LSTM FROM SCRATCH USING PYTORCH BY STATQUEST
With study case : predicting sales at day 5 of company A and B
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader


class LSTMbyHand(L.LightningModule):
    def __init__(self):
        super().__init__()
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # generate weights and bias
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    
    # do the math of LSTM
    def lstm_unit(self, input_value, long_memory, short_memory):
        # forget gate
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + (input_value * self.wlr2) + self.blr1) 
        # input gate
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1) 
        # candidate cell state
        potential_memory = torch.tanh((short_memory * self.wp1) + (input_value * self.wp2) + self.bp1)
        # update cell state
        updated_long_memory = (long_remember_percent * long_memory) + (potential_remember_percent * potential_memory)
        # output gate
        output_percent = torch.sigmoid((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        return ([updated_long_memory, updated_short_memory])

    # forward pass
    def forward(self, input):
        long_memory = 0
        short_memory = 0

        # input data
        day1 = input[0]
        day2 = input[1]
        day3 = input[2]
        day4 = input[3]

        # forward pass will update short_memory and long_memory for each day input
        long_memory, short_memory = self.lstm_unit(day1, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day2, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day3, long_memory, short_memory)
        long_memory, short_memory = self.lstm_unit(day4, long_memory, short_memory)

        return short_memory
    
    # optimizer
    def configure_optimizers(self):
        return Adam(self.parameters())
    
    # training step and calculate loss
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i)**2

        self.log('train_loss', loss)

        if (label_i == 0):
            self.log('out_0', output_i)
        else:
            self.log('out_1', output_i)

        return loss 
    
