from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 CNN block / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=20),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=256,
                      kernel_size=100),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1763584, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.flatten(x)
        logits = self.linear1(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 201, 201))
    # (1, 64, 44) is the shape of the signal which we obtain in dataset.py
