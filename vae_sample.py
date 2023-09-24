import time

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

import encoders_decorders.simple_ed as simple_ed

BATCH_SIZE = 100

trainval_data = MNIST("./sample_data",
                      train=True,
                      download=True,
                      transform=transforms.ToTensor())

train_size = int(len(trainval_data) * 0.8)
val_size = int(len(trainval_data) * 0.2)
train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(dataset=val_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0)

print("train data size: ", len(train_data))  # train data size:  48000
print("train iteration number: ", len(train_data) // BATCH_SIZE)  # train iteration number:  480
print("val data size: ", len(val_data))  # val data size:  12000
print("val iteration number: ", len(val_data) // BATCH_SIZE)  # val iteration number:  120

images, labels = next(iter(train_loader))
print("images_size:", images.size())  # images_size: torch.Size([100, 1, 28, 28])
print("label:", labels[:10])  # label: tensor([7, 6, 0, 6, 4, 8, 5, 2, 2, 3])


def criterion(predict, target, _ave, _log_dev):
    bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + _log_dev - _ave ** 2 - _log_dev.exp())
    return bce_loss + kl_loss


z_dim = 2
num_epochs = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = simple_ed.VAE(z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

history = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels": []}

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for i, (x, labels) in enumerate(train_loader):
        input_arr = x.to(device).view(-1, 28 * 28).to(torch.float32)
        output_arr, z, ave, log_dev = model(input_arr)

        history["ave"].append(ave)
        history["log_dev"].append(log_dev)
        history["z"].append(z)
        history["labels"].append(labels)
        loss = criterion(output_arr, input_arr, ave, log_dev)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print(f'Epoch: {epoch + 1}, loss: {loss: 0.4f}')
        history["train_loss"].append(loss)

    model.eval()
    loss = float("inf")
    with torch.no_grad():
        for i, (x, labels) in enumerate(val_loader):
            input_arr = x.to(device).view(-1, 28 * 28).to(torch.float32)
            output_arr, z, ave, log_dev = model(input_arr)

            loss = criterion(output_arr, input_arr, ave, log_dev)
            history["val_loss"].append(loss)

        print(f'Epoch: {epoch + 1}, val_loss: {loss: 0.4f}')

    scheduler.step()
end_time = time.time()
print("elapsed time: {0:.2f}".format(end_time - start_time))

train_loss_tensor = torch.stack(history["train_loss"])
train_loss_np = train_loss_tensor.to('cpu').detach().numpy().copy()
plt.plot(train_loss_np)
val_loss_tensor = torch.stack(history["val_loss"])
val_loss_np = val_loss_tensor.to('cpu').detach().numpy().copy()
plt.plot(val_loss_np)
plt.show()

ave_tensor = torch.stack(history["ave"])
log_var_tensor = torch.stack(history["log_dev"])
z_tensor = torch.stack(history["z"])
labels_tensor = torch.stack(history["labels"])
print(ave_tensor.size())  # torch.Size([9600, 100, 2])
print(log_var_tensor.size())  # torch.Size([9600, 100, 2])
print(z_tensor.size())  # torch.Size([9600, 100, 2])
print(labels_tensor.size())  # torch.Size([9600, 100])

ave_np = ave_tensor.to('cpu').detach().numpy().copy()
log_var_np = log_var_tensor.to('cpu').detach().numpy().copy()
z_np = z_tensor.to('cpu').detach().numpy().copy()
labels_np = labels_tensor.to('cpu').detach().numpy().copy()
print(ave_np.shape)  # (9600, 100, 2)
print(log_var_np.shape)  # (9600, 100, 2)
print(z_np.shape)  # (9600, 100, 2)
print(labels_np.shape)  # (9600, 100)

cmap_keyword = "tab10"
cmap = plt.get_cmap(cmap_keyword)

batch_num = 10
plt.figure(figsize=[10, 10])
for label in range(10):
    x = z_np[:batch_num, :, 0][labels_np[:batch_num, :] == label]
    y = z_np[:batch_num, :, 1][labels_np[:batch_num, :] == label]
    plt.scatter(x, y, color=cmap(label / 9), label=label, s=15)
    plt.annotate(label, xy=(np.mean(x), np.mean(y)), size=20, color="black")
plt.legend(loc="upper left")

batch_num = 9580
plt.figure(figsize=[10, 10])
for label in range(10):
    x = z_np[batch_num:, :, 0][labels_np[batch_num:, :] == label]
    y = z_np[batch_num:, :, 1][labels_np[batch_num:, :] == label]
    plt.scatter(x, y, color=cmap(label / 9), label=label, s=15)
    plt.annotate(label, xy=(np.mean(x), np.mean(y)), size=20, color="black")
plt.legend(loc="upper left")

# Reference: https://qiita.com/gensal/items/613d04b5ff50b6413aa0
