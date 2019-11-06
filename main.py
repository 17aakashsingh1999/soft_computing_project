from trainers import *
from models import *
from utils import *
from os import mkdir
import torch.optim as optim
from torch import nn

def main():
    inp = int(input('Enter choice:\n1. train DCNN new layers\n2. train DCNN complete\n3. train SDL layers\n4. train SDL + DCNN\n5. predict\n'))
    print('using device', device)
    try:
        mkdir('trained_models')
    except:
        pass
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
        net.load_state_dict(torch.load('trained_models/DCNN_partial'))

        data = load_dataset()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
        train_DCNN_complete(net=net, trainloader=data['train'], testloader=data['test'], optimizer=optimizer, criterion=criterion)
        print('training complete')

    elif inp == 3:
        # train SDL layers
        dcnn1 = DCNN()
        dcnn2 = DCNN()
        
        dcnn1.load_state_dict(torch.load('trained_models/DCNN_complete'))
        dcnn2.load_state_dict(torch.load('trained_models/DCNN_complete'))

        net = SDL()
        net.load_dcnn(dcnn1, dcnn2)
        net = net.to(device)

        data = load_dataset()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        train_SDL_partial(net=net, trainloader=data['train'], testloader=data['test'], optimizer=optimizer, criterion=criterion)
        print('training complete')
    elif inp == 4:
        # train SDL + DCNN
        dcnn1 = DCNN()
        dcnn2 = DCNN()
        
        dcnn1.load_state_dict(torch.load('trained_models/DCNN_complete'))
        dcnn2.load_state_dict(torch.load('trained_models/DCNN_complete'))

        net = SDL()
        net.load_state_dict(torch.load('trained_models/SDL_partial'))
        net.load_dcnn(dcnn1, dcnn2)
        
        net = net.to(device)
        
        data = load_dataset()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        train_SDL_complete(net=net, trainloader=data['train'], testloader=data['test'], optimizer=optimizer, criterion=criterion)
    elif inp == 5:
        # predict
        pass
    else:
        print('wrong input')


if __name__ == "__main__":
    main()