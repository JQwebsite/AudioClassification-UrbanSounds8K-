import torch

ResNet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')

ResNet18.conv1 = torch.nn.Conv2d(1,
                                 64,
                                 kernel_size=(7, 7),
                                 stride=(2, 2),
                                 padding=(3, 3),
                                 bias=False)

ResNet18.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)
