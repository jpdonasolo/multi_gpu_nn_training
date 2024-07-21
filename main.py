from mpi4py import MPI


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from model import CNNClassifier


# Seeds e cuda backend para fazer todas as runs com os mesmos resultados
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

is_master = MPI.COMM_WORLD.Get_rank() == 0


# Pega GPU se estiver habilitada e se tiver uma GPU por processo
# (Feito assim para facilitar teste do codigo, que foi desenvolvido
# no meu PC mas testado no kaggle)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    if size <= torch.cuda.device_count():
        device = device + f":{rank}"
    else:
        device = "cpu"

print(f"Rank: {rank}, Device: {device}")


def train_loop(model, opt, loss_func, train_loader):
    model.train()

    bsz = train_loader.batch_size
    
    assert bsz % size == 0, "Batch size must be divisible by number of workers"
    obs_per_wkr = bsz // size

    for X, y in train_loader:
        # Divide os dados entre os workers. Primeiro worker pega 
        # metade dos dados, segundo worker pega a outra metade
        X, y = X[rank * obs_per_wkr:(rank + 1) * obs_per_wkr], y[rank * obs_per_wkr:(rank + 1) * obs_per_wkr]
        X, y = X.to(device), y.to(device)

    
        opt.zero_grad()

        y_pred = model(X)
        loss = loss_func(y_pred, y)
        loss.backward()

        # Allreduce dos gradientes
        allreduce_params(model)

        opt.step()


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
    print(f"Test accuracy: {acc*100:.2f}%, test loss: {loss.item():.4f}")


def allreduce_params(model):
    # Tensores precisam estar no mesmo device
    model = model.to("cuda:0")

    for param in model.parameters():
        comm.allreduce(param.grad.data, op=MPI.SUM)
        param.grad.data /= size

    model = model.to(device)
            


# Apenas o master baixa os dados
train_data, test_data = None, None
if is_master:
    train_data = torchvision.datasets.MNIST(root="./dados", download=True, train=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(root="./dados", download=True, train=False, transform=torchvision.transforms.ToTensor())

train_data = comm.bcast(train_data, root=0)
test_data = comm.bcast(test_data, root=0)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)


# Broadcast do modelo para todos os workers para garantir mesma inicialização
model = None
if is_master:
    model = CNNClassifier()
    print(f"Modelo com {sum(p.numel() for p in model.parameters())} parâmetros")

model = comm.bcast(model, root=0)
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = nn.CrossEntropyLoss()

for epoch in range(10):
    if is_master:
        print(f"Epoch {epoch} ", end="")

    train_loop(model, opt, loss_func, train_loader)

    if is_master:
        test_loop(model, loss_func, test_loader)
