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
import argparse

from molxspec import utils


def cli(scan_hparams, prod_hparams):
    parser = argparse.ArgumentParser()
    parser.add_argument('--prod', action='store_true', default=False)
    args = parser.parse_args()
    if args.prod:
        print('+++ This is a *prod* run +++')
        setup_args = {
            #'scheduler': lambda *args, **kwargs: optim.lr_scheduler.OneCycleLR(*args, 1e-3, **kwargs),
            'scheduler': None,
            'swa': False,
            'n_epochs': 200,
            'save_model': True
        }
        hparams = prod_hparams
    else:
        print('+++ This is a scanning run +++')
        setup_args = {
            'save_model': True,
            'n_epochs': 50,
            'swa': False,
            'scheduler': None
        }
        hparams = scan_hparams
    return setup_args, args, hparams


def clear_device():
    torch.cuda.empty_cache()


def setup_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
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
            optimizer = optim.Adam,
            dataloader = DataLoader,
            device = None,
            scheduler = None,
            swa = False,
            save_model = False, 
            checkpoint = None
            ):
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.save_model = save_model
        self.swa = swa
        self.scheduler = scheduler
        self.train_data, self.test_data, self.validation_data = split_dataset(dataset)
        self.checkpoint = None if checkpoint is None else torch.load(checkpoint)

        self.train_loader = dataloader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = dataloader(self.test_data, batch_size=self.batch_size)
        self.device = setup_device() if device is None else device
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.optimizer = optimizer


    def _train_evaluate(self):
        torch.manual_seed(utils.RANDOM_SEED)
        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        if self.scheduler is not None:
            scheduler = self.scheduler(
                optimizer, 
                epochs=self.n_epochs, 
                steps_per_epoch=len(self.train_loader)
                )
        else:
            scheduler = None
        
        if self.checkpoint is not None:
            print('Loading checkpoint')
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        losses = {'train': {}, 'test': {}}
        state_to_save = {'loss': np.inf}
        for epoch in range(self.n_epochs):
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
                if scheduler is not None:
                    scheduler.step()

                losses['train'][epoch] += loss.item() / len(self.train_loader)

            for inputs, targets in self.test_loader:
                with torch.no_grad():
                    loss = criterion(
                            self.model(inputs),
                            targets.float().to(self.device)
                            )
                    losses['test'][epoch] += loss.item() / len(self.test_loader)
                    if True:#losses['test'][epoch] < state_to_save['loss']:
                        state_to_save['epoch'] = epoch
                        state_to_save['model_state_dict'] = self.model.state_dict()
                        state_to_save['model_kwargs'] = self.model.kwargs
                        if self.save_model:
                            state_to_save['optimizer_state_dict'] = optimizer.state_dict()
                        state_to_save['loss'] = losses['test'][epoch]
            print(f'Epoch: {epoch}: Loss train: [{losses["train"][epoch]}] | test: [{losses["test"][epoch]}]')

        if self.save_model:
            print(f'Epoch that was checkpointed: {state_to_save["epoch"]}')
            torch.save(state_to_save, f'{self.outdir}/best_checkpoint.pt')
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
