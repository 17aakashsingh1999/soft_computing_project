from trainers import *
from models import *
from utils import *

import torch.optim as optim
from torch import nn

def main():
    data = load_dataset()
    net = DCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_DCNN_complete(net, data['train'], data['test'], optimizer, criterion)

if __name__ == "__main__":
    main()