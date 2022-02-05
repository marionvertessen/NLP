import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, emb_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2e = nn.Embedding(input_size, emb_size)
        self.i2h = nn.Linear(emb_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(emb_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        embedded = self.i2e(input)

        combined = torch.cat((embedded, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, size):
        return torch.zeros(size, self.hidden_size)
