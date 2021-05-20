#!python3

import torch
import pathlib
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime
from torch.optim import Adam, lr_scheduler


class AvgMeter:
    """class for tracking the average of loss"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0.
        self.val = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0.
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _create_parent(fname):
    """ensure that the parent directory is created"""
    fpath = pathlib.Path(fname)
    dirpath = fpath.parent
    dirpath.mkdir(parents=True, exist_ok=True)


def _create_dir(dirname):
    """given a path to a directory, ensure that it is created"""
    dirpath = pathlib.Path(dirname)
    dirpath.mkdir(parents=True, exist_ok=True)


class ProgNet(nn.Module):
    """the neural network"""

    def __init__(self):

        super(ProgNet, self).__init__()

        # changing kernel_size to odd number to use same convolution
        self._block1 = nn.Sequential(
            nn.Conv1d(18, 20, (9,), padding=(4,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(20, 20, (9,), padding=(4,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(20, 1, (9,), padding=(4,)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(50, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
            nn.ReLU(inplace=True)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.zero_()
        return None

    def forward(self, x):
        return self._block1(x)


class PHMModel:
    """the model along with optimizer and methods for training"""

    def __init__(self, params, trainable):
        self.p = params
        self.trainable = trainable
        self._compile()

    def _compile(self):
        print('compiling PHM model...')
        self.model = ProgNet()
        if self.trainable:
            self.optim = Adam(self.model.parameters())

        self.loss = nn.MSELoss()

        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()
        self.p.cuda = self.use_cuda
        print('done compiling PHM model')

    def _print_params(self):
        """print the model parameter"""
        print('\nmodel parameters: ')
        param_dict = vars(self.p)
        for k, v in param_dict.items():
            print(f'{k} = {str(v)}')
        print('\n')

    def write_model(self, model=None, name='PHM'):
        """write a model to disk"""
        if model is None:
            model = self.model
        _create_dir(self.p.save_dir)
        fname = f'{self.p.save_dir}/{name}.pt'
        torch.save(model.state_dict(), fname)

    def write_stats(self, stats):
        """save stats as csv"""
        fpath = f'{self.p.save_dir}/n2n-stats.csv'
        array = np.array(stats)
        fmt = '%d,%f'
        header = 'epoch,train_loss'
        np.savetxt(fpath, array, fmt=fmt, header=header)

    def load_model(self, fname):
        print(f'loading checkpoint from: {fname}')
        if self.use_cuda:
            self.model.load_state_dict(torch.load(fname))
        else:
            self.model.load_state_dict(torch.load(fname, map_location='cpu'))
        return None

    def _on_epoch_end(self, stats, train_loss, epoch):
        """print info at the end of the epoch"""
        print(f'Train loss {train_loss:.6f} | ')

        # save checkpoint
        stats.append([epoch + 1, train_loss])
        if self.p.overwrite:
            name = 'PHM'
        else:
            name = f'PHM_{epoch + 1}_{train_loss:.6f}'
        print(f'saving checkpoint to {name}\n')

        self.write_model(name=name)
        self.write_stats(stats)

    def _on_training_end(self):
        self.write_model(name='final_model')

    def train(self, train_loader):
        self.model.train(True)
        self._print_params()

        # tracked stats
        stats = []
        train_start = datetime.now()

        # main training loop
        for epoch in range(self.p.num_epochs):
            print(f'EPOCH ({epoch + 1:d}/{self.p.num_epochs:d})')

            # training loss tracker
            loss_meter = AvgMeter()

            # minibatch SGD
            for source, target in tqdm(train_loader, desc='TRAIN'):
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()
                predicted = self.model(source.float())
                loss = self.loss(predicted, target.float())
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                # update the loss tracker
                loss_meter.update(loss.item())

            # save model and reset tracker
            self._on_epoch_end(stats, loss_meter.avg, epoch)
        self._on_training_end()
        train_elapsed = str(datetime.now() - train_start).split('.')
        print(f'training done, elapsed time: {train_elapsed[0]}\n')
