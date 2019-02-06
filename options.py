import argparse
import torch.nn as nn

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        self.parser.add_argument('--batch', type=int, nargs='?', default=32, help='batch size to be used')
        self.parser.add_argument('--lr', type=int, nargs='?', default=0.0001, help='learning rate')
        self.parser.add_argument('--epochs', type=int, nargs='?', default=300, help='total number of training episodes')
        self.parser.add_argument('--criterion', type=str, nargs='?', default=nn.L1Loss(), help='Loss Criterion to be used' )

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt



"""

options = options()

opts = options.parse()
batch = opts.batch
"""