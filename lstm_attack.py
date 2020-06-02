import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import hmms
import inference_attack
import sys

torch.manual_seed(1)

class BDPLSTM(nn.Module):

    def __init__(self, input_dim = 1, hidden_dim = 1, output_dim = 1):
        super(BDPLSTM, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)


    def forward(self, sanitized_db):
        lstm_out, _ = self.lstm(sanitized_db.view(1, sanitized_db.size()[0], 1))
        linear_out = self.linear(lstm_out)
        #apply softmax if going to use output of self.linear
        return lstm_out 


def LSTMAttacker(training_data, num_epochs=10):
    model = BDPLSTM()
    loss_function = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr =0.1)

    for epoch in range(num_epochs):
        for x, y in training_data:
            model.zero_grad()
            out = model(x)
            loss = loss_function(out, y.repeat(x.size()[0], 1).reshape(x.size()[0], -1))
            loss.backward()
            optimizer.step()
            # for param in model.parameters():
                # print(param)
        print(f"Loss after epoch {epoch} : {loss.item()}")

    cumulative_loss = 0
    with torch.no_grad():
        for x, y in list(training_data):
            print(model(x).float().round())
            # print(latent)
            cumulative_loss += loss_function(model(x).float().round(), y.repeat(x.size()[0], 1).reshape(x.size()[0], -1))

    return 1.-((cumulative_loss/len(training_data)).item())

num_hidden_states = 1
num_observations = 1
seq_len = 30

if __name__ == "__main__":
    eps = 0.5
    theta = 0.2
    print('eps:', eps, file=sys.stderr)
    dp_noise = inference_attack.diff_privacy_noise(eps)
    print('noise:', dp_noise, file=sys.stderr)
    print('Probability of not flipping state', 1-dp_noise, file=sys.stderr)

    model = inference_attack.symmetric_hmm(theta, dp_noise)
    latents, _ = model.generate_data((num_hidden_states, seq_len))
    attack_state = seq_len//2
    size = 4

    training_data_inputs = torch.zeros((num_hidden_states * (seq_len-(size)), size), dtype=torch.float)
    training_data_outputs = torch.zeros((num_hidden_states * (seq_len-(size)), 1),  dtype=torch.float)
    for idx, latent in enumerate(latents):
        for inner_idx in range(seq_len-size-1):
            training_data_inputs[idx*(seq_len-(size)) + inner_idx] = torch.tensor(latent[inner_idx:inner_idx+size])
            training_data_outputs[idx*(seq_len-(size)) + inner_idx] = torch.tensor(latent[inner_idx+size+1])
        
        # observations = []
        # for i in range(num_observations):
            # observations.append(inference_attack.emissions(latent, model.b))
        # training_data[1][idx] = torch.tensor(observations)


    training_data = list(zip(training_data_inputs, training_data_outputs))
    lstm_prob = LSTMAttacker(training_data) 
    print(lstm_prob)