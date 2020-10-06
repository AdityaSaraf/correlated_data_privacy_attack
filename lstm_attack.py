import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import hmms
import sys
from hmmlearn import hmm as skh
from tqdm import tqdm
import argparse
from math import exp
import math
from shapely.geometry import LineString, Point


def partitions(number, k):
    """
    Distribution of the folds
    Args:
        number: number of patients
        k: folds number
    """
    n_partitions = np.ones(k) * int(number / k)
    n_partitions[0 : (number % k)] += 1
    return n_partitions


def get_indices(n_splits, dataset_length):
    """
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    """
    l = partitions(dataset_length, n_splits)
    fold_sizes = l
    indices = np.arange(dataset_length).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size
        current = stop
        yield (indices[int(start) : int(stop)])


def k_folds(n_splits, dataset_length):
    """
    Generates folds for cross validation
    Args:
        n_splits: folds number
        dataset_length: length of dataset
    """
    indices = np.arange(dataset_length).astype(int)
    for test_idx in get_indices(n_splits, dataset_length):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx


class BDPLSTM(nn.Module):
    def __init__(self, hidden_dim=4, output_dim=1, embedding_dim=4):
        super(BDPLSTM, self).__init__()

        self.embedding = nn.Embedding(2, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Sigmoid()

    def forward(self, sanitized_db):
        x = self.embedding(sanitized_db)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # just keep last lstm output
        x = self.linear(x)
        prob = self.output(x)
        return prob

    # probs, targets shape: [batch_size, 1]
    def loss(self, probs, targets):
        return nn.functional.binary_cross_entropy(probs, targets.float())

    # probs, targets shape: [batch_size, 1]
    def accuracy(self, probs, targets):
        correct = (probs.round() == targets.float()).sum()
        return correct.float() / probs.size(0)


def LSTMAttacker(trainloader, testloader, logfile_name, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BDPLSTM()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    for epoch in range(num_epochs):
        train_correct = 0.0
        train_seqs = 0.0
        train_loss = 0.0
        model.train()
        tqdm_train = tqdm(trainloader, ncols=100, mininterval=1, ascii=True)
        for batch_idx, data in enumerate(tqdm_train):
            x, y = data[0].to(device), data[1].to(device)
            model.zero_grad()
            out = model(x)
            loss = model.loss(out, y)
            loss.backward()
            optimizer.step()
            correct = model.accuracy(out, y).item() * len(out)
            train_loss += loss.item()
            train_correct += correct
            train_seqs += len(x)
            tqdm_train.set_description_str(
                f"[Loss]: {train_loss / (batch_idx + 1):.4f} [Acc]: {train_correct / train_seqs:.4f}"
            )

        test_acc = 0.0
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                x, y = data[0].to(device), data[1].to(device)
                out = model(x)
                total += len(x)
                correct += (model.accuracy(out, y).item()) * len(out)
            test_acc = correct / total

        # print(
        #     f"Loss after epoch {epoch} : {train_loss / len(trainloader)}", file=sys.stderr)
        # print(
        #     f"Accuracy after epoch {epoch} : {train_correct / train_seqs}", file=sys.stderr)
        # print(
        #     f"Test accuracy after epoch {epoch} : {test_acc}", file=sys.stderr)

        # print(
        #     f"Loss after epoch {epoch} : {train_loss / len(trainloader)}", file=logfile_name)
        # print(
        #     f"Accuracy after epoch {epoch} : {train_correct / train_seqs}", file=logfile_name)
        # print(
        #     f"Test accuracy after epoch {epoch} : {test_acc}", file=logfile_name)

    return train_correct / train_seqs, test_acc
