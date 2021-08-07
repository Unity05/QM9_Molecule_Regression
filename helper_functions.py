import numpy as np


class AverageMeter:
    def __init__(self):
        self.losses = []

    def step(self, loss):
        self.losses.append(loss)

    def average(self):
        average_loss = np.mean(self.losses)

        return average_loss

    def std(self):
        std_loss = np.std(self.losses)

        return std_loss

    def clear(self):
        self.losses = []


class CosineLearningRateScheduler:
    def __init__(self, i_lr, n_batches_warmup, n_total_batches):
        self.i_lr = i_lr
        self.n_batches_warmup = n_batches_warmup
        self.current_batch = 0
        self.n_total_batches = n_total_batches

    def new_lr(self):
        if self.current_batch < self.n_batches_warmup:
            # learning rate warmup
            # starting with a too big learning rate may result in something unwanted (e.g. chaotic weights)
            lr = self.current_batch * (self.i_lr / self.n_batches_warmup)
        else:
            # cosine learning rate decay
            # (smoother than step learning rate decay)
            lr = self.i_lr * 0.5 * (1 + np.cos(((self.current_batch - self.n_batches_warmup) * np.pi) /
                                               self.n_total_batches))

        self.current_batch += 1

        return lr
