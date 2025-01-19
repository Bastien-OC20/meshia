import torch
import torch.nn as nn

class PointCloudDecoder(nn.Module):
    def __init__(self, input_dim=512, num_points=1024):
        super(PointCloudDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_points * 3)  # Chaque point est une coordonnée (x, y, z)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 3)  # Retourne un tableau de forme (num_points, 3)
