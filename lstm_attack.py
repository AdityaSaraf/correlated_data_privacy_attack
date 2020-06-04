import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hmms
import inference_attack
import sys

torch.manual_seed(1)

class BDPLSTM(nn.Module):

    def __init__(self, input_dim = 1, hidden_dim = 10, output_dim = 1):
        super(BDPLSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()


    # sanitized_db.shape = [batchsize, seq_len, input_dim]
    def forward(self, sanitized_db):
        lstm_out, _ = self.lstm(sanitized_db.view(1, sanitized_db.size()[0], 1))
        # [batchsize, seq_len, hidden_dim]
        lstm_out = lstm_out[:,-1,:] # just keep last lstm output
        linear_out = self.linear(lstm_out)
        prob = self.sigmoid(linear_out)
        return prob.reshape(1)

    # probs, targets shape: [batch_size, 1]
    def loss(self, probs, targets):
        return nn.functional.binary_cross_entropy(probs, targets)

    def accuracy(self, probs, targets):
        """
        Computes the accuracy, i.e number of correct predictions / N.
        :param probs: Raw predictions from the model of shape (N, 1)
        :param targets: True labels of shape (N, 1)
        :return: Accuracy as a scalar tensor.
        """
        correct = (probs.round() == targets).sum()
        return correct.float()/probs.size(0)    


# training_data = [(N, 4), (N, 1)]
def LSTMAttacker(training_data, num_epochs=20):
    model = BDPLSTM()
    optimizer = optim.Adam(model.parameters(), lr=7e-4)

    # last_loss = 1
    # unimproved_iters = 0
    # unimproved_tolerance = 10
    for epoch in range(num_epochs):
        # print(f"Epoch: {epoch}")
        train_correct = 0.
        train_seqs = 0.
        train_loss = 0.0
        for x, y in training_data:
            model.zero_grad()
            out = model(x)
            loss = model.loss(out, y) 
            loss.backward()
            optimizer.step()
            correct = model.accuracy(out, y).item()
            train_loss += loss.item()
            train_correct += correct
            train_seqs += 1 #len(x)

        print(f"Loss after epoch {epoch} : {train_loss / train_seqs}")
        print(f"Accuracy after epoch {epoch} : {train_correct / train_seqs}")
        # if (loss.item() < last_loss):
        #     unimproved_iters = 0
        # if (loss.item() > last_loss):
        #     unimproved_iters += 1
        # if (unimproved_iters > unimproved_tolerance):
        #     break
        # if (loss.item() <= 0.001):
        #     break

    return train_correct / train_seqs
    # cumulative_loss = 0
    # with torch.no_grad():
    #     for x, y in list(training_data):
    #         print(model(x).float().round())
    #         cumulative_loss += loss_function(model(x).float().round(), y.repeat(x.size()[0], 1).reshape(x.size()[0], -1))

    # return 1.-((cumulative_loss/len(training_data)).item())

num_hidden_states = 1
num_observations = 1
seq_len = 5000

if __name__ == "__main__":
    eps = 0.5
    theta = 0.2
    print('eps:', eps, file=sys.stderr)
    dp_noise = inference_attack.diff_privacy_noise(eps)
    print('noise:', dp_noise, file=sys.stderr)
    print('Probability of not flipping state (lower bound)', 1-dp_noise, file=sys.stderr)
    print(f'Upper bound: {1-theta}')

    model = inference_attack.symmetric_hmm(theta, dp_noise)
    latents, _ = model.generate_data((num_hidden_states, seq_len))
    size = 4

    training_data_inputs = torch.zeros((num_hidden_states * (seq_len-(size)), size), dtype=torch.float)
    training_data_outputs = torch.zeros((num_hidden_states * (seq_len-(size)), 1),  dtype=torch.float)
    # collecting data
    for idx, latent in enumerate(latents):
        sanitized = inference_attack.emissions(latent, model.b)
        transition_counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
        for i in range(len(latent) - 1):
            from_state = latent[i]
            to_state = latent[i+1]
            transition_counts[(from_state, to_state)] += 1

        q = transition_counts[(False, True)]/(transition_counts[(False, True)] + transition_counts[(False, False)])
        r = transition_counts[(True, False)]/(transition_counts[(True, False)] + transition_counts[(True, True)])
        print(f"q: {q}, r: {r}")
        for inner_idx in range(seq_len-size):
            training_data_inputs[idx*(seq_len-(size)) + inner_idx] = torch.tensor(sanitized[inner_idx:inner_idx+size])
            training_data_outputs[idx*(seq_len-(size)) + inner_idx] = torch.tensor(latent[inner_idx+size-1])
        
        # observations = []
        # for i in range(num_observations):
            # observations.append(inference_attack.emissions(latent, model.b))
        # training_data[1][idx] = torch.tensor(observations)
    

    training_data = list(zip(training_data_inputs, training_data_outputs))
    # trainloader = torch.utils.data.DataLoader(training_data, batch_size=b,
    #                                     shuffle=True, num_workers=2)
    lstm_accuracy = LSTMAttacker(training_data)
     
    print(f"LSTM accuracy: {lstm_accuracy}")