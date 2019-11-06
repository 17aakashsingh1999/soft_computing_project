from trainers import *
from models import *
from utils import *

import torch.optim as optim
from torch import nn

def main():
    inp = int(input('Enter choice:\n1. train DCNN new layers\n2. train DCNN complete\n3. train SDL layers\n4. train SDL + DCNN\n5. predict\n'))

    if inp == 1:
        # train DCNN new layers
        net = DCNN().to(device)
        data = load_dataset()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
        train_DCNN_partial(net=net, trainloader=data['train'], testloader=data['test'], optimizer=optimizer, criterion=criterion)
        print('training complete')

    elif inp == 2:
        # train DCNN complete
        net = DCNN().to(device)
        data = load_dataset()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
        train_DCNN_complete(net=net, trainloader=data['train'], testloader=data['test'], optimizer=optimizer, criterion=criterion)
        print('training complete')
    elif inp == 3:
        # train SDL layers
        train_SDL_partial()
    elif inp == 4:
        # train SDL + DCNN
        train_SDL_complete()
    elif inp == 5:
        # predict
        pass
    else:
        print('wrong input')


if __name__ == "__main__":
    main()