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
        correct = (probs.round() == targets).sum()
        return correct.float()/probs.size(0)


# training_data = [(N, 4), (N, 1)]
def LSTMAttacker(trainloader, testloader, num_epochs=15):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BDPLSTM()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print('Starting training')
    for epoch in range(num_epochs):
        train_correct = 0.
        train_seqs = 0.
        train_loss = 0.0
        model.train()
        for batch_idx, data in enumerate(trainloader):
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
            trainloader.set_description_str(
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

        print(f"Loss after epoch {epoch} : {train_loss / len(trainloader)}")
        print(f"Accuracy after epoch {epoch} : {train_correct / train_seqs}")
        print(f"Test accuracy after epoch {epoch} : {test_acc}")

        return train_correct / train_seqs, test_acc

    print('Training done')


num_hidden_states = 1
num_observations = 1
seq_len = 30000
num_folds = 5

if __name__ == "__main__":
    eps = 0.5
    theta = 0.1
    print('eps:', eps, file=sys.stderr)
    dp_noise = inference_attack.diff_privacy_noise(eps)
    print('noise:', dp_noise, file=sys.stderr)
    print('Probability of not flipping state (lower bound)',
          1-dp_noise, file=sys.stderr)
    print(f'Upper bound: {1-theta}')

    hmm = inference_attack.symmetric_hmm(theta, dp_noise)
    latents, _ = hmm.generate_data((num_hidden_states, seq_len))
    size = 20

    # latents_test, _ = hmm.generate_data((num_hidden_states, test_len))

    inputs = torch.zeros(
        (num_hidden_states * (seq_len-(size)), size), dtype=torch.long)
    outputs = torch.zeros(
        (num_hidden_states * (seq_len-(size)), 1),  dtype=torch.long)
    # collecting data
    # test_data_inputs = torch.zeros(
    #     (num_hidden_states * (test_len-(size)), size), dtype=torch.long)
    # test_data_outputs = torch.zeros(
    #     (num_hidden_states * (test_len-(size)), 1),  dtype=torch.long)
    for idx, latent in enumerate(latents):
        sanitized = inference_attack.emissions(latent, hmm.b)

        # FOR TESTING PURPOSES ONLY
        # np.save("latent", latent)
        # np.save("sanitized", sanitized)
        # latent = np.load("latent.npy")
        # sanitized = np.load("sanitized.npy")

        # _, predictions = hmm.viterbi(sanitized[size//2:-size//2])
        # count = np.sum(predictions == latent[size//2:-size//2])
        # print(f'Viterbi Accuracy (knows parameters): {count/seq_len}')

        # More accurate, but underflows (should just implement separate algo which maximizes only the transition probabilities)
        # rho = dp_noise
        # theta_hats = []
        # for i in range(seq_len//750):
        #     a = np.ones((2, 2))
        #     a = a / np.sum(a, axis=1)
        #     emissions = np.array([[1-rho, rho], [rho, 1-rho]])
        #     pi = np.array((0.5, 0.5))
        #     a, b = BaumWelch.baum_welch(
        #         sanitized[i*750:(i+1)*750], a, emissions, pi, n_iter=100)
        #     print(f'a:{a} \nb:{b}')
        #     theta_hats.append((a[0, 1] + a[1, 0])/2)
        # theta_hat = np.average(theta_hats)
        # p_hat = np.array([[1-theta_hat, theta_hat], [theta_hat, 1-theta_hat]])
        # print(f'P_hat = {p_hat}')

        # custom_model: hmms.DtHMM = hmms.DtHMM(p_hat, emissions, pi)
        # _, predictions = custom_model.viterbi(sanitized[size//2:-size//2])
        # count = np.sum(predictions == latent[size//2:-size//2])
        # print(
        #     f'Viterbi Accuracy (learns parameters with custom BW): {count/seq_len}')

        for inner_idx in range(seq_len-size-2):
            inputs[idx*(seq_len-(size)) + inner_idx] = torch.tensor(
                sanitized[inner_idx:inner_idx+size])
            outputs[idx*(seq_len-(size)) +
                    inner_idx] = torch.tensor(latent[inner_idx+size//2])

    # for idx, latent in enumerate(latents_test):
    #     sanitized = inference_attack.emissions(latent, hmm.b)
    #     for inner_idx in range(test_len-size-2):
    #         test_data_inputs[idx*(test_len-(size)) + inner_idx] = torch.tensor(
    #             sanitized[inner_idx:inner_idx+size])
    #         test_data_outputs[idx*(test_len-(size)) +
    #                           inner_idx] = torch.tensor(latent[inner_idx+size//2])

    data = list(zip(inputs, outputs))
    print(len(data))
    # training_data = list(zip(inputs, outputs))
    # test_data = list(zip(test_data_inputs, test_data_outputs))

    for train_idx, test_idx in k_folds(5, len(data)):
        print(len(train_idx), len(test_idx))
        training_data = torch.utils.data.Subset(data, indices=train_idx)
        test_data = torch.utils.data.Subset(data, indices=test_idx)

        trainloader = torch.utils.data.DataLoader(
            training_data, batch_size=6, shuffle=True, pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=6, shuffle=False, pin_memory=True)
        train_acc, test_acc = LSTMAttacker(
            tqdm(trainloader, ncols=100), tqdm(testloader, ncols=100))
        print(f"Train accuracy: {train_acc}, test accuracy: {test_acc}")
