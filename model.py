from torch import nn

class DNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.feed = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 3),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.feed(x)