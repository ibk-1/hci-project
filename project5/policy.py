import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*5*5, 64), nn.ReLU(),
            nn.Linear(64, 4)  # logits for 4 actions
        )

    def forward(self, x):  # x: [B,6,5,5]
        return self.net(x)

    def pi(self, x):
        return torch.softmax(self.forward(x), dim=-1)




