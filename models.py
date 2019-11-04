import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from utils import device
from torchsummary import summary

class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        self.net = resnet18(pretrained=True)

        for param in self.net.parameters():
            param.require_grad = False
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 256)

        self.layer1 = nn.Linear(256, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 16)
        self.fc     = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.net(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.softmax(self.fc(x))
        return x

    def forward_extracted_features(self, x):
        x = F.relu(self.net(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

class SDL(nn.Module):
    def __init__(self, dcnn1, dcnn2):
        super(SDL, self).__init__()
        self.dcnn1 = dcnn1
        self.dcnn2 = dcnn2

        self.layer1 = nn.Linear(32, 16)
        self.layer2 = nn.Linear(16, 4)
        self.fc     = nn.Linear(4, 2)

    def forward(self, x1, x2):
        x1 = self.dcnn1.forward_extracted_features(x1)
        x2 = self.dcnn2.forward_extracted_features(x2)

        x = torch.cat((x1, x2), dim=1).reshape((-1, 32))
        # x = torch.rand((1,32)).to(device)
        print('hello', x1.shape, x2.shape, x.shape)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.fc(x))
        return x





if __name__ == "__main__":
    summary(DCNN().to(device), input_size=(3, 224, 224))
    summary(resnet18(pretrained=True).to(device), input_size=(3, 224, 224))
    summary(SDL(DCNN(), DCNN()).to(device), input_size=[(3, 224, 224), (3, 224, 224)])