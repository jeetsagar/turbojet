#!python3

"""training the model in pytorch"""

from torchmodel import PHMModel
from torchdata import load_testdata
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='test model')

    # data parameters
    parser.add_argument('--testdata', help='path to test data', default='../../../data_set/N-CMAPSS_DS02-006.h5')
    parser.add_argument('--load-file', help='model location', default='../../checkpoints/final_model.pt')
    parser.add_argument('--output', help='model output', default='../../output/output.npz')

    # model options
    parser.add_argument('--units', help='units to use', action='append', nargs='*', type=int, default=[])
    parser.add_argument('--sequence', help='use sequence model', action='store_true')
    parser.add_argument('-b', '--batch-size', help='batch size', default=128, type=int)

    # testing options
    parser.add_argument('--cuda', help='use cuda', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    params = parse_args()
    test_loader = load_testdata(params)
    model = PHMModel(params, trainable=False)
    model.load_model(params.load_file)
    model.test(test_loader)
