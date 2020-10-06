from math import exp
import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import LineString, Point
import torch
import lstm_attack
import hmms
import argparse
import sys


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

# Currently only returns the middle calculation, where neither rho_0 nor rho_1 are 0.5
# Returns (rho_0, rho_1)


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
        intersect_top = line2.intersection(line_top)
        intersect_right = line1.intersection(line_right)
    else:
        intersect_top = line1.intersection(line_top)
        intersect_right = line2.intersection(line_right)
    intersect = line1.intersection(line2)
    pts = [intersect, intersect_top, intersect_right]
    exp_noise = np.zeros((3,))
    for i, pt in enumerate(pts):
        if isinstance(pt, Point):
            exp_noise[i] = pt.x * r/(q+r) + pt.y * q/(q+r)
        else:
            exp_noise[i] = float('inf')
    min_pt = pts[np.argmin(exp_noise)]
    return min_pt.x, min_pt.y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_for_loop", dest="no_for_loop",
                        action="store_true")
    parser.add_argument("--eps", dest="eps", type=float, default=0.2)
    parser.add_argument("--data", dest="data", type=str,
                        default="data/normal/N8")
    parser.add_argument("--output", dest="output_prefix", type=str,
                        default="real_data_output")
    args = parser.parse_args()

    output_prefix = args.output_prefix
    no_for_loop = args.no_for_loop
    data = args.data

    if no_for_loop:
        supplied_eps = args.eps
        lst_eps = [supplied_eps]
        output = f"{output_prefix}_{eps}.csv"
        output_log = f"{output_prefix}_{eps}.log"
    else:
        lst_eps = np.arange(1, 5.5, 0.5)
        output = f"{output_prefix}.csv"
        output_log = f"{output_prefix}.log"

    log_fd = open(output_log, "w+")

    lines = []
    with open('data/normal/N8', 'r') as f:
        lines = f.read().splitlines()
        lines = [item.split(' ')[1] for item in lines]
    data = np.array(lines, dtype=float)

    avg = np.average(data)
    # state 0 (aka False) indicates rest state
    binary_dat = np.array([item >= avg for item in data])
    transition_counts = {(True, True): 0, (True, False): 0,
                         (False, True): 0, (False, False): 0}

    for i in range(len(binary_dat) - 1):
        from_state = binary_dat[i]
        to_state = binary_dat[i+1]
        transition_counts[(from_state, to_state)] += 1

    q = transition_counts[(
        False, True)]/(transition_counts[(False, True)] + transition_counts[(False, False)])
    r = transition_counts[(
        True, False)]/(transition_counts[(True, False)] + transition_counts[(True, True)])
    print('(q, r):', (q, r), file=sys.stderr)
    print('(q, r):', (q, r), file=log_fd)

    with open(output, "a+") as f:
        f.write(f"eps, viterbi_accuracy, lstm_accuracy\n")
    for eps in lst_eps:
        rho_0, rho_1 = min_exp_noise(q, r, eps)
        print('eps:', eps, '(rho_0, rho_1):', (rho_0, rho_1), file=log_fd)
        print('eps:', eps, '(rho_0, rho_1):', (rho_0, rho_1), file=sys.stderr)
        data = np.array(binary_dat, dtype=bool)
        for i, val in enumerate(data):
            if val:  # 1 state
                if np.random.rand() < rho_1:
                    data[i] = not val
            else:  # 0 state
                if np.random.rand() < rho_0:
                    data[i] = not val

        size = 100
        all_data_inputs = torch.zeros(
            (data.size-(size), size), dtype=torch.long)
        all_data_outputs = torch.zeros(
            (data.size - (size), 1), dtype=torch.long)

        latent = binary_dat.astype(np.int_)
        sanitized = data.astype(np.int_)

        print(
            f'SB acc:{np.sum(latent == sanitized)/latent.shape[0]}', file=sys.stderr)
        print(
            f'SB acc:{np.sum(latent == sanitized)/latent.shape[0]}', file=log_fd)

        p = np.array([[1-q, q], [r, 1-r]])
        pi = np.array((0.5, 0.5))
        emissions = np.array([[1-rho_0, rho_0], [rho_1, 1-rho_1]])

        hmm_model = hmms.DtHMM(p, emissions, pi)

        _, predictions = hmm_model.viterbi(sanitized)
        count = np.sum(predictions == latent)
        viterbi_accuracy = count/predictions.shape[0]
        print(
            f'Viterbi Acc: {viterbi_accuracy}', file=sys.stderr)
        print(
            f'Viterbi Acc: {viterbi_accuracy}', file=log_fd)

        for inner_idx in range(data.size-size):
            all_data_inputs[inner_idx] = torch.tensor(
                sanitized[inner_idx:inner_idx+size])
            all_data_outputs[inner_idx] = torch.tensor(
                latent[inner_idx+size//2])

        all_data = list(zip(all_data_inputs, all_data_outputs))
        num_folds = 5
        lstm_avg_acc = 0.
        fold_idx = 0
        for train_idx, test_idx in lstm_attack.k_folds(num_folds, len(all_data)):
            print(f'fold: {fold_idx}', file=sys.stderr)
            print(f'fold: {fold_idx}', file=log_fd)
            fold_idx += 1
            training_data = torch.utils.data.Subset(
                all_data, indices=train_idx)
            test_data = torch.utils.data.Subset(all_data, indices=test_idx)

            trainloader = torch.utils.data.DataLoader(
                training_data, batch_size=6, shuffle=True, pin_memory=True)
            testloader = torch.utils.data.DataLoader(
                test_data, batch_size=6, shuffle=False, pin_memory=True)
            train_acc, test_acc = lstm_attack.LSTMAttacker(
                trainloader, testloader, log_fd)
            lstm_avg_acc += test_acc
            print(
                f"Train accuracy: {train_acc}, test accuracy: {test_acc}", file=sys.stderr)
            print(
                f"Train accuracy: {train_acc}, test accuracy: {test_acc}", file=log_fd)
        lstm_avg_acc /= num_folds
        with open(output, "a+") as f:
            f.write(f"{eps}, {viterbi_accuracy}, {lstm_avg_acc}\n")

    log_fd.close()
