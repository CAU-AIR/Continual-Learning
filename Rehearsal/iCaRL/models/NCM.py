import numpy as np
from math import ceil

import torch
from torch import nn

class NearestClassMean(nn.Module):
    def __init__(self, input_shape, num_classes, alpha=0.9, device='cuda'):
        super(NearestClassMean, self).__init__()

        # NCM parameters
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.alpha = alpha
        self.device = device

        # setup weights for NCM
        self.muK = torch.zeros((self.num_classes, self.input_shape)).to(self.device)
        self.prev_muK = torch.zeros((self.num_classes, self.input_shape)).to(self.device)

        self.cK = torch.zeros(self.num_classes).to(self.device)

    @torch.no_grad()
    def fit(self, x, y):
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        # update class means
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1

    @torch.no_grad()
    def find_dists(self, A, B):
        M, d = B.shape
        with torch.no_grad():
            B = torch.reshape(B, (M, 1, d))  # reshaping for broadcasting
            square_sub = torch.mul(A - B, A - B)  # square all elements
            dist = torch.sum(square_sub, dim=2)
        return -dist    # why use minus

    @torch.no_grad()
    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)
        scores = self.find_dists(self.muK, X)

        # mask off predictions for unseen classes
        not_visited_ix = torch.where(self.cK == 0)[0]
        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        # return predictions or probabilities
        if not return_probas:
            return scores
        else:
            return torch.softmax(scores, dim=1)

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y):
        # fit NCM one example at a time
        for x, y in zip(batch_x, batch_y):
            self.fit(x.cpu(), y.view(1, ))

    @torch.no_grad()
    def train_(self, feature, target):
        batch_x_feat = feature.to(self.device)
        self.fit_batch(batch_x_feat, target)

    @torch.no_grad()
    def evaluate_(self, batch_x):
        num_samples = len(batch_x)
        probabilities = torch.empty((num_samples, self.num_classes)).to(self.device)
        start = 0

        # for x in batch_x:
        # probas = self.predict(x, return_probas=True)
        probas = self.predict(batch_x, return_probas=True)
        end = start + probas.shape[0]
        probabilities[start:end] = probas
        start = end
        
        return probabilities
    
    @torch.no_grad()
    def update_mean(self, task):
        for idx in task:
            self.muK[idx, :] = (self.alpha * self.prev_muK[idx, :]) + ((1-self.alpha)*self.muK[idx, :])
            self.prev_muK[idx, :] = self.muK[idx, :]
            self.cK[idx] = 0