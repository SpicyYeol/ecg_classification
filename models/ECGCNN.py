import torch.nn as nn
import torch.nn.functional as F


class ECGCNN(nn.Module):
    def __init__(self, num_classes, ecg_data_length):
        super(ECGCNN, self).__init__()

        self.ecg_data_length = ecg_data_length

        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (self.ecg_data_length // 8),
                             128)  # assuming input length is divisible by 8 after pooling
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * (self.ecg_data_length // 8))  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)#F.softmax(x, dim=1)#F.sigmoid(x)#F.softmax(x, dim=1)
