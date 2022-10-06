from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 CNN block / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=2), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=2), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=2), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=2), nn.ReLU())
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=23836032, out_features=5000)
        self.linear2 = nn.Linear(in_features=5000, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear1(x)
        logits = self.linear2(logits)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    cnn = CNNNetwork()
    summary(cnn.cuda(), (1, 853, 200))
    # (1, 64, 44) is the shape of the signal which we obtain in dataset.py