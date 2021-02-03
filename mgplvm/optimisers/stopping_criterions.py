import numpy as np


class LossMarginStop():

    def __init__(self, loss_margin=0, stop_after=10):
        '''
        loss_margin [optional] : float, default=0
            loss margin tolerated for training progress, can be negative (require improvement every iteration)

        stop_iters [optional]: int, default=10
            maximum number of training iterations above loss margin tolerated before stopping
        '''
        self.lowest_loss = np.inf
        self.stop_ = 0
        self.loss_margin = loss_margin
        self.stop_after = stop_after

    def __call__(self, model, i, loss_val):
        if loss_val <= self.lowest_loss + self.loss_margin:
            self.stop_ = 0
        else:
            self.stop_ += 1

        if loss_val < self.lowest_loss:
            self.lowest_loss = loss_val

        return (self.stop_ > self.stop_after)
