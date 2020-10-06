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
from lstm_attack import *


num_hidden_states = 1
num_observations = 1
seq_len = 25000
num_folds = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_for_loop", dest="no_for_loop",
                        action="store_true")
    parser.add_argument("--eps", dest="eps", type=float, default=None)
    parser.add_argument("--output", dest="output_prefix", type=str,
                        default="synthetic_output")
    args = parser.parse_args()

    output_prefix = args.output_prefix
    no_for_loop = args.no_for_loop
    eps = args.eps
    no_for_loop = no_for_loop or eps

    if (no_for_loop):
        supplied_eps = args.eps
        lst_eps = [supplied_eps]
        output = f"{output_prefix}_{eps}.csv"
        output_log = f"{output_prefix}_{eps}.log"
    else:
        lst_eps = np.arange(1, 5.5, 0.5)
        output = f"{output_prefix}.csv"
        output_log = f"{output_prefix}.log"

    log_fd = open(output_log, "w+")

    with open(output, "a+") as f:
        f.write(f"eps, viterbi_accuracy, lstm_accuracy\n")

    q, r = 0.08937981353871098, 0.10924092409240924
    transitions = np.array([[1-q, q],
                            [r, 1-r]])
    pi = [0.5, 0.5]
    for eps in lst_eps:
        # eps = 2
        print('eps:', eps, file=sys.stderr)
        print('eps:', eps, file=log_fd)
        rho_0, rho_1 = min_exp_noise(q, r, eps)
        emissions = np.array([[1-rho_0, rho_0],
                              [rho_1, 1-rho_1]])
        print('rho_0, rho_1:', rho_0, rho_1, file=sys.stderr)
        print('rho_0, rho_1:', rho_0, rho_1, file=log_fd)

        hmm = hmms.DtHMM(transitions, emissions, pi)
        latent, sanitized = hmm.generate_data((num_hidden_states, seq_len))
        latent = latent[0]
        sanitized = sanitized[0]
        size = 100

        inputs = torch.zeros(
            (num_hidden_states * (seq_len-(size)), size), dtype=torch.long)
        outputs = torch.zeros(
            (num_hidden_states * (seq_len-(size)), 1),  dtype=torch.long)

        print(
            f'SB acc:{np.sum(latent == sanitized)/latent.shape[0]}', file=sys.stderr)
        print(
            f'SB acc:{np.sum(latent == sanitized)/latent.shape[0]}', file=log_fd)

        _, predictions = hmm.viterbi(sanitized[size//2:-size//2])
        count = np.sum(predictions == latent[size//2:-size//2])
        viterbi_accuracy = count/predictions.shape[0]
        print(
            f'Viterbi Accuracy (knows parameters): {viterbi_accuracy}', file=sys.stderr)
        print(
            f'Viterbi Accuracy (knows parameters): {viterbi_accuracy}', file=log_fd)
        print(f'SB Attacker: {np.sum(latent==sanitized)/len(latent)}')

        for inner_idx in range(seq_len-size-2):
            inputs[inner_idx] = torch.tensor(
                sanitized[inner_idx:inner_idx+size])
            outputs[inner_idx] = torch.tensor(latent[inner_idx+size//2])

        data = list(zip(inputs, outputs))
        print(len(data))

        num_folds = 5
        lstm_avg_acc = 0.
        for train_idx, test_idx in k_folds(num_folds, len(data)):
            print(len(train_idx), len(test_idx))
            training_data = torch.utils.data.Subset(data, indices=train_idx)
            test_data = torch.utils.data.Subset(data, indices=test_idx)

            trainloader = torch.utils.data.DataLoader(
                training_data, batch_size=6, shuffle=True, pin_memory=True)
            testloader = torch.utils.data.DataLoader(
                test_data, batch_size=6, shuffle=False, pin_memory=True)
            train_acc, test_acc = LSTMAttacker(trainloader, testloader, log_fd)
            lstm_avg_acc += test_acc
            print(
                f"Train accuracy: {train_acc}, test accuracy: {test_acc}", file=sys.stderr)
            print(
                f"Train accuracy: {train_acc}, test accuracy: {test_acc}", file=log_fd)

        lstm_avg_acc /= num_folds
        lstm_avg_acc = 0
        with open(output, "a+") as f:
            f.write(f"{eps}, {viterbi_accuracy}, {lstm_avg_acc}\n")

    log_fd.close()
