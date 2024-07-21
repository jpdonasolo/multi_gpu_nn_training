from mpi4py import MPI


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.functional as F
import torchvision

from modelo import CNNClassifier



def test_loop(model, loss_func, test_loader):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_func(y_pred, y)

            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    acc = correct / total
    if is_master:
        print(f"Accuracy: {acc}")

def train_loop(model, opt, loss_func, train_loader):
    model.train()

    #batch size 
    bsz = train_loader.batch_size
    obs_per_wkr = bsz // size

    for X, y in train_loader:
        X, y = X[rank * obs_per_wkr:(rank + 1) * obs_per_wkr], y[rank * obs_per_wkr:(rank + 1) * obs_per_wkr]
        X, y = X.to(device), y.to(device)
    
        opt.zero_grad()

        y_pred = model(X)
        loss = loss_func(y_pred, y)
        loss.backward()

        allreduce_params(model)
        comm.Barrier()

        opt.step()

    if is_master:
        print(f"Loss: {loss.item()}")


def allreduce_params(model):
    for param in model.parameters():
        comm.Allreduce(MPI.IN_PLACE, param.grad.data, op=MPI.SUM)
        param.grad.data /= size
            


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

is_master = MPI.COMM_WORLD.Get_rank() == 0


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    assert size <= torch.cuda.device_count()
    device = device + f":{rank}"

print(f"Rank: {rank}, Device: {device}")


train_data = torchvision.datasets.MNIST(root="./dados", download=True, train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root="./dados", download=True, train=False, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


model = None
if is_master:
    model = CNNClassifier(n_filters_1=16, kernel_size_1=3, n_filters_2=32, kernel_size_2=3, n_hidden=128, dropout=True)

model = comm.bcast(model, root=0)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()

for epoch in range(10):
    train_loop(model, opt, loss_func, train_loader)
    
    # if is_master:
    #     test_loop(model, loss_func, test_loader)
    
    comm.Barrier()
