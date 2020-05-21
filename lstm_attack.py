import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)

class BDPLSTM(nn.Module):

    def __init__(self, input_dim = 1, output_dim = 1):
        super(BDPLSTM, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True)


    def forward(self, sanitized_db):
        lstm_out, _ = self.lstm(sanitized_db.view(sanitized_db.size()[0], sanitized_db.size()[1], 1))
        return lstm_out 

def LSTMAttacker(training_data):
    model = BDPLSTM()
    loss_function = torch.nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr =0.1)

    for epoch in range(15):
        for latent, sanitized in training_data:
            model.zero_grad()
            out = model(sanitized)
            loss = loss_function(out, latent.repeat(sanitized.size()[0], 1, 1).reshape(sanitized.size()[0], -1, 1))
            loss.backward()
            optimizer.step()

    cumulative_loss = 0
    with torch.no_grad():
        for latent, sanitized in list(training_data):
            cumulative_loss += loss_function(model(sanitized), latent.repeat(sanitized.size()[0], 1, 1).reshape(sanitized.size()[0], -1, 1))

    return 1.-((cumulative_loss/len(training_data)).item())