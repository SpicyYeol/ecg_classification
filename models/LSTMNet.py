import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTMNet(nn.Module):
    def __init__(self, num_classes):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 128).to(x.device)  # 2 layers, batch size, hidden size
        c0 = torch.zeros(2, x.size(0), 128).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]  # take the last output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x