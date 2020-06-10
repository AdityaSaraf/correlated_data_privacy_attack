import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import hmms
import inference_attack
import BaumWelch
import sys
from hmmlearn import hmm as skh
from tqdm import tqdm
import argparse
from math import exp
import math
from shapely.geometry import LineString, Point


def A(q, r, eps):
    er = exp(eps)*r
    root = math.sqrt((q - er)**2 + 4*exp(eps)*(1-q)*(1-r))
    return abs(q - er) * root


def B(q, r, eps):
    return (q-exp(eps)*r)**2 + 2 * exp(eps) * (1-q)*(1-r)


def C(q, r, eps):
    return 2*(1-q)**2 * exp(eps)


# Represents A', the result of switching q and r in A
def A_alt(q, r, eps):
    return A(r, q, eps)


# Represents B', the result of switching q and r in B
def B_alt(q, r, eps):
    return B(r, q, eps)


# Represents C', the result of switching q and r in B
def C_alt(q, r, eps):
    return C(r, q, eps)


def min_exp_noise(q, r, eps):
    e = exp(eps)
    a = A(q, r, eps)
    b = B(q, r, eps)
    c = C(q, r, eps)
    a_ = A_alt(q, r, eps)
    b_ = B_alt(q, r, eps)
    c_ = C_alt(q, r, eps)
    if (e >= q/r):
        line1 = LineString([(1-(b+a)/(2*c), 0.5), (0.5, c/(2*(b+a)))])
    else:  # e < q/r
        line1 = LineString([(1-(b-a)/(2*c), 0.5), (0.5, c/(2*(b-a)))])
    if (e >= r/q):
        line2 = LineString([(c_/(2*(b_+a_)), 0.5), (0.5, 1-(b_+a_)/(2*c_))])
    else:  # e < r/q
        line2 = LineString([(c_/(2*(b_-a_)), 0.5), (0.5, 1-(b_-a_)/(2*c_))])

    (x_1, y_1), (x_2, y_2) = line1.coords
    slope1 = (y_2-y_1)/(x_2 - x_1)
    (x_1, y_1), (x_2, y_2) = line2.coords
    slope2 = (y_2-y_1)/(x_2 - x_1)
    line_top = LineString([(0, 0.5), (0.5, 0.5)])
    line_right = LineString([(0.5, 0), (0.5, 0.5)])
    if abs(slope2) > abs(slope1):
        intsect_top = line2.intersection(line_top)
        intsect_right = line1.intersection(line_right)
    else:
        intsect_top = line1.intersection(line_top)
        intsect_right = line2.intersection(line_right)
    intsect = line1.intersection(line2)
    pts = [intsect, intsect_top, intsect_right]
    exp_noise = np.zeros((3,))
    for i, pt in enumerate(pts):
        if isinstance(pt, Point):
            exp_noise[i] = pt.x * r/(q+r) + pt.y * q/(q+r)
        else:
            exp_noise[i] = float('inf')
    # return intsect.x, intsect.y
    print(exp_noise)
    min_pt = pts[np.argmin(exp_noise)]
    return min_pt.x, min_pt.y


def partitions(number, k):
    '''
    Distribution of the folds
    Args:
        number: number of patients
        k: folds number
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions


def get_indices(n_splits, dataset_length):
    '''
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    l = partitions(dataset_length, n_splits)
    fold_sizes = l
    indices = np.arange(dataset_length).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop = current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])


def k_folds(n_splits, dataset_length):
    '''
    Generates folds for cross validation
    Args:
        n_splits: folds number
        dataset_length: length of dataset
    '''
    indices = np.arange(dataset_length).astype(int)
    for test_idx in get_indices(n_splits, dataset_length):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx


class BDPLSTM(nn.Module):

    def __init__(self, hidden_dim=4, output_dim=1, embedding_dim=4):
        super(BDPLSTM, self).__init__()

        self.embedding = nn.Embedding(2, embedding_dim)
        # , bidirectional=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Sigmoid()

    # sanitized_db.shape = [batchsize, seq_len, input_dim]

    def forward(self, sanitized_db):
        x = self.embedding(sanitized_db)
        x, _ = self.lstm(x)
        # [batchsize, seq_len, hidden_dim]
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
        return correct.float()/probs.size(0)


# training_data = [(N, 4), (N, 1)]
def LSTMAttacker(trainloader, testloader, logfile_name, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BDPLSTM()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    print('Starting training')
    for epoch in range(num_epochs):
        train_correct = 0.
        train_seqs = 0.
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
                f'[Loss]: {train_loss / (batch_idx + 1):.4f} [Acc]: {train_correct / train_seqs:.4f}')

        test_acc = 0.
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                x, y = data[0].to(device), data[1].to(device)
                out = model(x)
                total += len(x)
                correct += (model.accuracy(out, y).item()) * len(out)
            test_acc = correct/total

        print(
            f"Loss after epoch {epoch} : {train_loss / len(trainloader)}", file=sys.stderr)
        print(
            f"Accuracy after epoch {epoch} : {train_correct / train_seqs}", file=sys.stderr)
        print(
            f"Test accuracy after epoch {epoch} : {test_acc}", file=sys.stderr)

        print(
            f"Loss after epoch {epoch} : {train_loss / len(trainloader)}", file=logfile_name)
        print(
            f"Accuracy after epoch {epoch} : {train_correct / train_seqs}", file=logfile_name)
        print(
            f"Test accuracy after epoch {epoch} : {test_acc}", file=logfile_name)

    return train_correct / train_seqs, test_acc


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
    transitions = np.array([[1-q,q],
                            [r,1-r]])
    pi = [0.5, 0.5]
    for eps in lst_eps:
        # eps = 2
        print('eps:', eps, file=sys.stderr)
        print('eps:', eps, file=log_fd)
        rho_0, rho_1 = min_exp_noise(q, r, eps)
        emissions = np.array([[1-rho_0,rho_0],
                             [rho_1,1-rho_1]])
        print('rho_0, rho_1:', rho_0, rho_1, file=sys.stderr)
        print('rho_0, rho_1:', rho_0, rho_1, file=log_fd)

        hmm = hmms.DtHMM(transitions, emissions, pi)
        latent, sanitized = hmm.generate_data((num_hidden_states, seq_len))
        latent = latent[0]
        sanitized = sanitized[0]
        size = 120

        inputs = torch.zeros(
            (num_hidden_states * (seq_len-(size)), size), dtype=torch.long)
        outputs = torch.zeros(
            (num_hidden_states * (seq_len-(size)), 1),  dtype=torch.long)

        # latent = latents[0]
        # sanitized = inference_attack.emissions(latent, hmm.b)
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
