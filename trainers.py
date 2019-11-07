import torch
from models import DCNN, SDL

from utils import create_sdl_dataset, device

def train_DCNN_partial(net, trainloader, testloader, optimizer, criterion, n_epochs=50):
    best_score = 0
    net.freeze_resnet()
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if correct > best_score:
            torch.save(net.state_dict(), 'trained_models/DCNN_partial')
            best_score = correct
    
        print('Accuracy of the network: %d %%' % (100 * correct / total))

    net.unfreeze_resnet()


def train_DCNN_complete(net, trainloader, testloader, optimizer, criterion, n_epochs=50):
    best_score = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print('epoch', epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if correct > best_score:
            torch.save(net.state_dict(), 'trained_models/DCNN_complete')
            best_score = correct
        
        print('Accuracy of the network: %d %%' % (100 * correct / total))

def train_SDL_partial(net, trainloader, testloader, optimizer, criterion, n_epochs=50):
    best_score = 0
    net.dcnn1.freeze_complete()
    net.dcnn2.freeze_complete()

    trainloader = create_sdl_dataset(trainloader)
    testloader = create_sdl_dataset(testloader)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs1, inputs2, labels = data
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images1, images2, labels = data
                images1 = images1.to(device)
                images2 = images2.to(device)
                labels = labels.to(device)
                outputs = net(images1, images2)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if correct > best_score:
            dcnn1 = net.get_dcnn1()
            dcnn2 = net.get_dcnn2()

            net.unload_dcnn()
            torch.save(net.state_dict(), 'trained_models/SDL_partial')

            net.load_dcnn(dcnn1, dcnn2)
            best_score = correct

        print('Accuracy of the network: %d %%' % (100 * correct / total))
    
    net.dcnn1.unfreeze_complete()
    net.dcnn2.unfreeze_complete()

def train_SDL_complete(net, trainloader, testloader, optimizer, criterion, n_epochs=50):
    best_score = 0
    trainloader = create_sdl_dataset(trainloader)
    testloader = create_sdl_dataset(testloader)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs1, inputs2, labels = data
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs1, inputs2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images1, images2, labels = data
                images1= images1.to(device)
                images2 = images2.to(device)
                labels = labels.to(device)
                outputs = net(images1, images2)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if correct > best_score:
            dcnn1 = net.get_dcnn1()
            dcnn2 = net.get_dcnn2()

            net.unload_dcnn()

            torch.save(net.state_dict(), 'trained_models/SDL_complete')
            torch.save(dcnn1.state_dict(), 'trained_models/SDL_DCNN1')
            torch.save(dcnn2.state_dict(), 'trained_models/SDL_DCNN2')
            
            net.load_dcnn(dcnn1, dcnn2)
            best_score = correct

        print('Accuracy of the network: %d %%' % (
            100 * correct / total))

def train_DCNN_finetune(net, trainloader, testloader, optimizer, criterion, name, n_epochs=50):
    net.freeze_except_last()
    best_score = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print('epoch', epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if correct > best_score:
            torch.save(net.state_dict(), 'trained_models/' + name)
            best_score = correct
        
        print('Accuracy of the network: %d %%' % (100 * correct / total))
    net.unfreeze_complete()