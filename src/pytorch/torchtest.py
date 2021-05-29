#!python3

"""training the model in pytorch"""

from torchmodel import PHMModel
from torchdata import load_traindata
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='train noise2noise model')

    # data parameters
    parser.add_argument('--testdata', default='../../../data_set/N-CMAPSS_DS02-006.h5')
    parser.add_argument('--units', action='append', nargs='*', type=int, default=[])
    parser.add_argument('--sequence', help='use sequence model', action='store_true')
    parser.add_argument('-b', '--batch-size', default=128, type=int)

    # training options
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    params = parse_args()
    test_loader = load_traindata(params)
    model = PHMModel(params, True)
    model.test(test_loader)
