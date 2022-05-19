from cProfile import label
from model import RobustModel
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
from torch.nn import ModuleList
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'
PATH = "./model.pt"

learning_rate = 0.0009
batch_size = 32


train_data_svhn = np.load('train_data_svhn.npy').astype(np.float32)
for i in range(len(train_data_svhn)):
    train_data_svhn[i] = train_data_svhn[i].astype(np.float32)/255
train_data_svhn = np.reshape(train_data_svhn, (-1, 1, 28, 28))
labels_svhn = np.load('train_label_svhn.npy')

train_data_mnist = np.load('train_data.npy').astype(np.float32)
for i in range(len(train_data_mnist)):
    train_data_mnist[i] = train_data_mnist[i].astype(np.float32)/255
train_data_mnist = np.reshape(train_data_mnist, (-1, 1, 28, 28))
labels_mnist = np.load('train_label.npy')

train_data_etc = np.load('train_data_etc.npy').astype(np.float32)
for i in range(len(train_data_etc)):
    train_data_etc[i] = train_data_etc[i].astype(np.float32)/255
train_data_etc = np.reshape(train_data_etc, (-1, 1, 28, 28))
labels_etc = np.load('train_label_etc.npy')

train_data = np.append(train_data_svhn, train_data_mnist, axis = 0)
train_data = np.append(train_data, train_data_etc, axis = 0)
labels = np.append(labels_svhn, labels_mnist)
labels = np.append(labels, labels_etc)


train_data_mnist = []
for i in range(len(train_data)):
    train_data_mnist.append([train_data[i], labels[i]])


split_ratio = 0.2
train2_num = int(len(train_data_mnist)*(1-split_ratio))
test_num = len(train_data_mnist)-train2_num
print(train2_num, test_num)
train_set, test_set = torch.utils.data.random_split(train_data_mnist, [train2_num, test_num])

train_num = int(train2_num * (1-split_ratio))
val_num = train2_num-train_num
print(train_num, val_num)
train_set, val_set = torch.utils.data.random_split(train_set, [train_num, val_num])

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
dev_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)


#test phase
def test(data_loader, model):
    model.eval()
    n_predict = 0
    n_correct = 0
    with torch.no_grad():
        for X, Y in tqdm.tqdm(data_loader):
            Y = Y.type(torch.LongTensor)
            X = X.cuda()
            Y = Y.cuda()
            y_hat = model(X)
            y_hat.argmax()

            _, predicted = torch.max(y_hat, 1)

            n_predict += len(predicted)
            n_correct += (Y == predicted).sum()
    accuracy = n_correct / n_predict
    print(f"Accuracy : {accuracy} ()")


#train phase
model = RobustModel().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=5e-5)

best_model = None
epochs = 100
train_cost_list = []
dev_cost_list = []
min_dev_cost = 10
for epoch in range(epochs):
    model.train()
    cost = 0.0
    n_batches = 0
    
    for X, Y in tqdm.tqdm(train_loader):
        Y = Y.type(torch.LongTensor)
        X = X.cuda()
        Y = Y.cuda()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        cost += loss.item()
        n_batches += 1
    cost /= n_batches
    print('[Epoch : {:>3}   cost = {:>.9}]'.format(epoch + 1, cost))
    torch.save(model.state_dict(), PATH)

    model.eval()
    n_batches = 0
    dev_cost = 0.0
    for X, Y in tqdm.tqdm(dev_loader):
        Y = Y.type(torch.LongTensor)
        X = X.cuda()
        Y = Y.cuda()
        outputs = model(X)
        loss = criterion(outputs, Y)
        dev_cost += loss.item()
        n_batches += 1
    dev_cost /= n_batches
    print('              devcost = {:>.9}]'.format(dev_cost))

    train_cost_list.append(cost)
    dev_cost_list.append(dev_cost)

    if dev_cost < min_dev_cost:
        min_dev_cost = dev_cost
        torch.save(model.state_dict(), "./best_model.pt")
        best_model = model

x = range(1,epochs+1)
plt.plot(x, train_cost_list, 'r', label = "train")
plt.plot(x, dev_cost_list, 'b', label = "dev")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.legend()
plt.show()
test(test_loader, best_model)
print('Finished Training')

