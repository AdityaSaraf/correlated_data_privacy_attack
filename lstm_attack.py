import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import hmms
import inference_attack
import BaumWelch
import sys

class BDPLSTM(nn.Module):

    def __init__(self, hidden_dim = 4, output_dim = 1, embedding_dim = 4):
        super(BDPLSTM, self).__init__()

        self.embedding = nn.Embedding(2, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)#, bidirectional=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Sigmoid()


    # sanitized_db.shape = [batchsize, seq_len, input_dim]
    def forward(self, sanitized_db):
        x = self.embedding(sanitized_db)
        x, _ = self.lstm(x)
        # [batchsize, seq_len, hidden_dim]
        x = x[:,-1,:] # just keep last lstm output
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
def LSTMAttacker(trainloader, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BDPLSTM()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(num_epochs):
        train_correct = 0.
        train_seqs = 0.
        train_loss = 0.0
        for data in trainloader:
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

        print(f"Loss after epoch {epoch} : {train_loss / len(trainloader)}")
        print(f"Accuracy after epoch {epoch} : {train_correct / train_seqs}")

    return train_correct / train_seqs

num_hidden_states = 1
num_observations = 1
seq_len = 10000

if __name__ == "__main__":
    eps = 0.5
    theta = 0.1
    print('eps:', eps, file=sys.stderr)
    dp_noise = inference_attack.diff_privacy_noise(eps)
    print('noise:', dp_noise, file=sys.stderr)
    print('Probability of not flipping state (lower bound)', 1-dp_noise, file=sys.stderr)
    print(f'Upper bound: {1-theta}')

    hmm = inference_attack.symmetric_hmm(theta, dp_noise)
    latents, _ = hmm.generate_data((num_hidden_states, seq_len))
    size = 20

    training_data_inputs = torch.zeros((num_hidden_states * (seq_len-(size)), size), dtype=torch.long)
    training_data_outputs = torch.zeros((num_hidden_states * (seq_len-(size)), 1),  dtype=torch.long)
    # collecting data
    for idx, latent in enumerate(latents):
        sanitized = inference_attack.emissions(latent, hmm.b)
        
        # FOR TESTING PURPOSES ONLY
        # np.save("latent", latent)
        # np.save("sanitized", sanitized)
        # latent = np.load("latent.npy")
        # sanitized = np.load("sanitized.npy")
        torch.manual_seed(0)
        np.random.seed(0)
        
        _, predictions = hmm.viterbi(sanitized)
        count = np.sum(predictions == latent)
        print(f'Viterbi Accuracy (knows parameters): {count/seq_len}')
        
        # More accurate, but underflows (should just implement separate algo which maximizes only the transition probabilities)

        rho = dp_noise
        theta_hats = []
        for i in range(seq_len//750):
            a = np.ones((2, 2))
            a = a / np.sum(a, axis=1)
            emissions = np.array([[1-rho,rho],[rho,1-rho]])
            pi = np.array((0.5, 0.5))
            a, b = BaumWelch.baum_welch(sanitized[i*750:(i+1)*750], a, emissions, pi, n_iter=100)
            print(f'a:{a} \nb:{b}')
            theta_hats.append((a[0, 1] + a[1,0])/2)
        theta_hat = np.average(theta_hats)
        p_hat = np.array([[1-theta_hat, theta_hat], [theta_hat, 1-theta_hat]])
        print(f'P_hat = {p_hat}')

        custom_model: hmms.DtHMM = hmms.DtHMM(p_hat, emissions, pi)
        _, predictions = custom_model.viterbi(sanitized)
        count = np.sum(predictions == latent)
        print(f'Viterbi Accuracy (learns parameters with custom BW): {count/seq_len}')

        for inner_idx in range(seq_len-size-2):
            training_data_inputs[idx*(seq_len-(size)) + inner_idx] = torch.tensor(sanitized[inner_idx:inner_idx+size])
            training_data_outputs[idx*(seq_len-(size)) + inner_idx] = torch.tensor(latent[inner_idx+size//2])
        
        # observations = []
        # for i in range(num_observations):
            # observations.append(inference_attack.emissions(latent, model.b))
        # training_data[1][idx] = torch.tensor(observations)
    

    training_data = list(zip(training_data_inputs, training_data_outputs))
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=16,
                                        shuffle=True, num_workers=4)
    lstm_accuracy = LSTMAttacker(trainloader)
    print(f"LSTM accuracy: {lstm_accuracy}")