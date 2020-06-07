from math import exp
import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import LineString, Point
import torch
import inference_attack
import lstm_attack
import hmms
import BaumWelch

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
        line1 = LineString([ (1-(b+a)/(2*c), 0.5), (0.5, c/(2*(b+a))) ])
    else: # e < q/r
        line1 = LineString([ (1-(b-a)/(2*c), 0.5), (0.5, c/(2*(b-a))) ])
    if (e >= r/q):
        line2 = LineString([ (c_/(2*(b_+a_)),0.5), (0.5,1-(b_+a_)/(2*c_)) ])
    else: # e < r/q
        line2 = LineString([ (c_/(2*(b_-a_)),0.5), (0.5,1-(b_-a_)/(2*c_)) ])
    
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
    # return intsect.x, intsect.y
    exp_noise = np.fromiter(map(lambda pt: pt.x * r/(q+r) + pt.y * q/(q+r), pts), dtype=float)
    print(exp_noise)
    min_pt = pts[np.argmin(exp_noise)]
    return min_pt.x, min_pt.y

if __name__ == "__main__":
    lines = []
    with open('data/normal/N10', 'r') as f:
        lines = f.read().splitlines()
        lines = [item.split(' ')[1] for item in lines]
    data = np.array(lines, dtype=float)
    print(len(data))
    # data = data[0:len(data):2]
    avg = np.average(data)
    binary_dat = np.array([item >= avg for item in data]) # state 0 (aka False) indicates rest state
    transition_counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}

    for i in range(len(binary_dat) - 1):
        from_state = binary_dat[i]
        to_state = binary_dat[i+1]
        transition_counts[(from_state, to_state)] += 1

    q = transition_counts[(False, True)]/(transition_counts[(False, True)] + transition_counts[(False, False)])
    r = transition_counts[(True, False)]/(transition_counts[(True, False)] + transition_counts[(True, True)])
    print('(q, r):', (q,r))
    print('Original sum: ', np.sum(binary_dat))
    # q, r = 0.02, 0.02
    eps = 2.0
    rho_0, rho_1 = min_exp_noise(q, r, eps)
    # print(min_exp_noise(0.35, 0.4, 0.5))
    print('eps:', eps, '(rho_0, rho_1):', (rho_0, rho_1))
    data = np.array(binary_dat, dtype=bool)
    for i, val in enumerate(data):
        if val: # 1 state
            if np.random.rand() < rho_1:
                data[i] = not val
        else: # 0 state
            if np.random.rand() < rho_0:
                data[i] = not val
    print('sum: ', np.sum(data))
    print(f"changed in {np.sum(data == binary_dat)} locations")

    size = 120
    seq_len = int((data.size-size)* 0.8)
    test_len = (data.size-size) - seq_len

    all_data_inputs = torch.zeros((data.size-(size), size), dtype=torch.long)
    all_data_outputs = torch.zeros((data.size- (size), 1), dtype = torch.long) 

    training_data_inputs = torch.zeros(((seq_len-(size)), size), dtype=torch.long)
    training_data_outputs = torch.zeros(((seq_len-(size)), 1),  dtype=torch.long)
    # collecting data
    test_data_inputs = torch.zeros((test_len-(size), size), dtype=torch.long)
    test_data_outputs = torch.zeros((test_len-(size), 1),  dtype=torch.long)
    
    latent = binary_dat.astype(np.int_)
    sanitized = data.astype(np.int_)

    print(f'Correlation ignorant attacker success prob:{np.sum(latent == sanitized)/latent.shape[0]}')

    p = np.array([[1-q, q], [r, 1-r]])
    pi = np.array((0.5, 0.5))
    emissions = np.array([[1-rho_0,rho_0],[rho_1,1-rho_1]])
    
    custom_model_knows_params = hmms.DtHMM(p, emissions, pi)

    _, predictions = custom_model_knows_params.viterbi(sanitized)

    viterbi_test_start = (data.size//2) - 10000
    viterbi_test_end = (data.size//2) + 10000
    count = np.sum(predictions[viterbi_test_start:viterbi_test_end] == latent[viterbi_test_start:viterbi_test_end])
    print(f'Viterbi Accuracy (knows parameters): {count/(viterbi_test_end - viterbi_test_start)}')
    count = np.sum(predictions == latent)
    print(f'Viterbi Overall Accuracy (knows parameters): {count/(predictions.shape[0])}')
        
    # transition_01 = []
    # transition_10 = []

    # for i in range(data.size//300):
    #     a = np.ones((2, 2))
    #     a = a / np.sum(a, axis=1)
    #     pi = np.array((0.5, 0.5))
    #     emissions = np.array([[1-rho_0,rho_0],[rho_1,1-rho_1]])
    #     a, b = BaumWelch.baum_welch(sanitized[i*300:(i+1)*300], a, emissions, pi, n_iter=1000)
    #     print(f'a:{a} \nb:{b}')
    #     transition_01.append(a[0,1])
    #     transition_10.append(a[1,0])
    
    # q_hat = np.average(transition_01)
    # r_hat = np.average(transition_10)

    # p_hat = np.array([[1-q_hat, q_hat], [r_hat, 1-r_hat]])
    # print(f'P_hat = {p_hat}')

    # custom_model: hmms.DtHMM = hmms.DtHMM(p_hat, emissions, pi)
    # _, predictions = custom_model.viterbi(sanitized)
    # count = np.sum(predictions[seq_len+size//2:-size//2] == latent[seq_len+size//2:-size//2])
    # print(f'Viterbi Accuracy (learns parameters with custom BW): {count/(test_len - size)}')

    for inner_idx in range(data.size-size):
        all_data_inputs[inner_idx] = torch.tensor(sanitized[inner_idx:inner_idx+size])
        all_data_outputs[inner_idx] = torch.tensor(latent[inner_idx+size//2])

    train_dataset, test_dataset = torch.utils.data.random_split(list(zip(all_data_inputs, all_data_outputs)), [seq_len, test_len])
    
    # for inner_idx in range(seq_len-size):
    #     training_data_inputs[inner_idx] = torch.tensor(sanitized[inner_idx:inner_idx+size])
    #     training_data_outputs[inner_idx] = torch.tensor(latent[inner_idx+size//2])
    
    # for inner_idx in range(test_len-size):
    #     test_data_inputs[inner_idx] = torch.tensor(sanitized[inner_idx:inner_idx+size])
    #     test_data_outputs[inner_idx] = torch.tensor(latent[inner_idx+size//2])

    # training_data = list(zip(training_data_inputs, training_data_outputs))
    # test_data = list(zip(test_data_inputs, test_data_outputs))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    train_acc, test_acc = lstm_attack.LSTMAttacker(trainloader, testloader)
    print(f"Train accuracy: {train_acc}, test accuracy: {test_acc}")
    print(transition_counts)