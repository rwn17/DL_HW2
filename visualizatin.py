import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class CustomResnet():
    def __init__(self, resnet_in):
        self.resnet = resnet_in

    def myforward(self, xb):
        #propogate to the layer before fc
        x = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(self.resnet.conv1(xb))))
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        # x = x.view(x.size(0), -1)
        return x

def valid(model, valid_loader,criterion):
    model.train(False)
    total_loss = 0.0
    total_correct = 0
    numpy_array = np.ones(2048,dtype='float32')
    label_array = np.ones(1,dtype='int')
    for inputs, labels in valid_loader:
        # stack X
        inputs = inputs.to(device)
        this_numpy = myresnet.myforward(inputs).cpu().detach().numpy()
        # this_numpy = this_numpy.reshape(1,-1)
        numpy_array = np.vstack([numpy_array,np.squeeze(this_numpy)])
        # print('X shape is:', numpy_array.shape)
        # stack y
        this_label = labels.numpy()
        label_array = np.vstack([label_array,(this_label)[:,None]])
        labels = labels.to(device)
        # calculate acc on valid set
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
    epoch_loss = total_loss / len(valid_loader.dataset)
    epoch_acc = total_correct.double() / len(valid_loader.dataset)
    numpy_array = numpy_array[1:numpy_array.size]
    label_array = label_array[1:label_array.size]
    return epoch_loss, epoch_acc.item(), numpy_array, label_array

if __name__ == '__main__':
    model = torch.load('./models/best_modelA.pt')
    data_dir = "../hw2_dataset/"
    inupt_size = 224
    batch_size = 36
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    myresnet = CustomResnet(model)

    criterion = nn.CrossEntropyLoss()
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=inupt_size, batch_size=batch_size)
    valid_loss, valid_acc, X, y = valid(model, valid_loader,criterion)
    print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
    print(" T-SNE begin")
    tsne = TSNE(n_components=2,random_state=501)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    y = y.flatten()
    for i in range(X_norm.shape[0]):
        color = plt.cm.Set1(y[i]/12)
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=color)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/t-sne.png')
    plt.show()