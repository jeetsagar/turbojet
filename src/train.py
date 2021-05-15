#!python3

from model import PHMModel
from dataset import load_traindata
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='train noise2noise model')

    # data parameters
    parser.add_argument('--traindata', default='../../data_set/N-CMAPSS_DS02-006.h5')
    parser.add_argument('--units', action='extend', nargs='*', type=int, default=[])
    parser.add_argument('--save-dir', help='model directory', default='../checkpoints')
    parser.add_argument('--overwrite', help='overwrite old models', action='store_true')

    # training hyperparameters
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-e', '--num-epochs', default=20, type=int)

    # training options
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    # write weights and biases info as text
    parser.add_argument('--writeinfo', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    train_loader = load_traindata(params)
    model = PHMModel(params, True)
    model.train(train_loader)
