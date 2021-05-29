#!python3

"""training the model in pytorch"""

from torchmodel import PHMModel
from torchdata import load_traindata
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='train model')

    # data parameters
    parser.add_argument('--traindata', help='path to training data', default='../../../data_set/N-CMAPSS_DS02-006.h5')
    parser.add_argument('--units', help='units used for training', action='append', nargs='*', type=int, default=[])
    parser.add_argument('--save-dir', help='model directory', default='../../checkpoints')
    parser.add_argument('--overwrite', help='overwrite old models', action='store_true', default=False)

    parser.add_argument('--load-file', help='model location', default='../../checkpoints/ProgNet.pt')

    parser.add_argument('--features-last', help='features in data', default=False, action='store_true')

    # training hyperparameters
    parser.add_argument('-b', '--batch-size', help='batch size', default=128, type=int)
    parser.add_argument('-e', '--num-epochs', help='number of epochs', default=20, type=int)

    # model options
    parser.add_argument('--sequence', help='use sequence model', action='store_true', default=False)
    parser.add_argument('--restore', help='restore pre-trained model', action='store_true', default=False)

    # training options
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    params = parse_args()
    train_loader = load_traindata(params)
    model = PHMModel(params, trainable=True)
    if params.restore:
        model.load_model(params.load_file)
        print(f'done loading checkpoint: {params.load_file}')
    model.train(train_loader)
