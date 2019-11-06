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
        
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, 256)

        self.layer1 = nn.Linear(256, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 16)
        self.fc     = nn.Linear(16, 2)

    def freeze_complete(self):
        for param in self.parameters():
            param.require_grad = False
    
    def unfreeze_complete(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_resnet(self):
        for param in self.net.parameters():
            param.requires_grad = False
    
    def unfreeze_resnet(self):
        for param in self.net.parameters():
            param.requires_grad = True

    def freeze_except_last(self):
        self.freeze_complete();
        for param in self.fc.parameters():
            param.requires_grad = True

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
    def __init__(self):
        super(SDL, self).__init__()
        self.fc     = nn.Linear(32, 2)
    
    def load_dcnn(self, dcnn1, dcnn2):
        self.set_dcnn1(dcnn1)
        self.set_dcnn2(dcnn2)
    
    def unload_dcnn(self):
        del self.dcnn1
        del self.dcnn2

    def set_dcnn1(self, dcnn1):
        self.dcnn1 = dcnn1
    
    def set_dcnn2(self, dcnn2):
        self.dcnn2 = dcnn2

    def get_dcnn1(self):
        return self.dcnn1
    
    def get_dcnn2(self):
        return self.dcnn2
    
    def forward(self, x1, x2):
        x1 = self.dcnn1.forward_extracted_features(x1)
        x2 = self.dcnn2.forward_extracted_features(x2)

        x = torch.cat((x1, x2), dim=1).reshape((-1, 32))
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.fc(x))
        return x



if __name__ == "__main__":
    summary(DCNN().to(device), input_size=(3, 224, 224))
    summary(resnet18(pretrained=True).to(device), input_size=(3, 224, 224))
    sdl = SDL()
    sdl.load_dcnn(DCNN(), DCNN())
    summary(sdl.to(device), input_size=[(3, 224, 224), (3, 224, 224)])