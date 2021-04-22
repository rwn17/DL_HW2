import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import BalancedSoftmaxLoss
from torch.optim import lr_scheduler

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def train_model(model,train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=20):

    def train(model, train_loader, scheduler, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            # print(scheduler.get_lr()[0])
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        print(scheduler.get_lr()[0])
        train_loss, train_acc = train(model, train_loader,scheduler, optimizer, criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        scheduler.step()
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, './models/best_modelB_fine.pt')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## about model
    num_classes = 10

    ## about data
    data_dir = "../hw2_dataset/" ## You need to specify the data_dir first
    inupt_size = 224
    batch_size = 128

    ## about training
    num_epochs = 200
    lr = 0.001

    ## model initialization
    # task = '1-Large-Scale'
    task = '4-Long-Tailed'
    # task = '4-Long-Tailed'
    if task == '1-Large-Scale':
        model = models.model_A(num_classes=num_classes)
        criterion = nn.CrossEntropyLoss()
    elif task == '2-Medium-Scale':
        model = models.model_B(num_classes=num_classes)
        # criterion = nn.CrossEntropyLoss()
    elif task == '4-Long-Tailed':
        model = models.model_C(num_classes=num_classes)
        LossJsonPath = './cls_freq/4-Long-Tailed.json'
        criterion = BalancedSoftmaxLoss.create_loss(LossJsonPath)
        # criterion = nn.CrossEntropyLoss()

    # choose NO. of GPU, if more than 2, use 2   
    gpu = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    if gpu > 1:
        print(torch.cuda.device_count())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model.to(device), [0,1])
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=inupt_size, batch_size=batch_size, task=task)

    ## optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # my best lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, [80,120,160], gamma=0.2)

    ## train model
    train_model(model,train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)
