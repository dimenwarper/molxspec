import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import bitsandbytes.optim as boptim

import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

import utils

def clear_device():
    torch.cuda.empty_cache()


def setup_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device

def split_dataset(dataset):
    train_size = int(len(dataset) * 0.8)
    test_size = int((len(dataset) - train_size) * 0.5)
    valid_size = len(dataset) - (train_size + test_size)
    return torch.utils.data.random_split(dataset, [train_size, test_size, valid_size])


class TrainingSetup:
    def __init__(
            self,
            model,
            dataset,
            outdir,
            n_epochs = 100,
            lr = 3e-4,
            batch_size = 300,
            optimizer = boptim.Adam8bit,
            dataloader = DataLoader
            ):
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_data, self.test_data, self.validation_data = split_dataset(dataset)

        self.train_loader = dataloader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = dataloader(self.test_data, batch_size=self.batch_size)
        self.device = setup_device()
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.optimizer = optimizer


    def _train_evaluate(self):
        torch.manual_seed(utils.RANDOM_SEED)
        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)

        losses = {'train': {}, 'test': {}}
        state_to_save = {'loss': np.inf}
        for epoch in tqdm(range(self.n_epochs), desc='Epoch', leave=False):
            losses['train'][epoch] = 0
            losses['test'][epoch] = 0
            for inputs, targets in self.train_loader:
                optimizer.zero_grad()
                loss = criterion(
                        self.model(inputs),
                        targets.to(self.device)
                        )
                loss.backward()
                optimizer.step()

                losses['train'][epoch] += loss.item() / len(self.train_loader)

            for inputs, targets in self.test_loader:
                with torch.no_grad():
                    loss = criterion(
                            self.model(inputs),
                            targets.float().to(self.device)
                            )
                    losses['test'][epoch] += loss.item() / len(self.test_loader)
                    if losses['test'][epoch] < state_to_save['loss']:
                        state_to_save['epoch'] = epoch
                        state_to_save['model_state_dict'] = self.model.state_dict()
                        #state_to_save['optimizer_state_dict'] = optimizer.state_dict()
                        state_to_save['loss'] = losses['test'][epoch]

        #torch.save(state_to_save, f'{self.outdir}/best_checkpoint.pt')
        return losses


    def _plot_curves(self, curves):
        plt.figure()
        for name, loss in curves.items():
            plt.plot(list(loss.values()), label=f'{name}')
            plt.yscale('log')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
        plt.savefig(f'{self.outdir}/learning_curves.png')


    def _save_stats(self, curves):
        pd.DataFrame(curves).to_csv(f'{self.outdir}/learning_curves.tsv')


    def train(self):
        clear_device()
        self.curves = self._train_evaluate()
        self._plot_curves(self.curves)
        self._save_stats(self.curves)
