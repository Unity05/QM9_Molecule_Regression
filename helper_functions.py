import numpy as np


class AverageMeter:
    def __init__(self):
        self.losses = []

    def step(self, loss):
        self.losses.append(loss)

    def average(self):
        average_loss = np.mean(self.losses)
        self.losses = []

        return average_loss
