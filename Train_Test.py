import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from  dataloader import train_loader, test_loader
from  MultiModalEnhancer import newnet
from Train_Test import Fusion_train_model, Fusion_test_model
import torch
from torch import nn

Epoch = 100
batch_size = 2

def Fusion_train_model(net, trainloader, optimizer, device):

    net.train()
    train_loss = 0
    train_acc = 0

    for x1, x2, label in tqdm(trainloader):
        optimizer.zero_grad()
        x1, x2, label= x1.to(device), x2.to(device), label.to(device)

        output = net(x1, x2)
        output = output.repeat(batch_size, 1)
        loss = F.cross_entropy(output.float(), label.long())
        loss.backward(retain_graph=True)


        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            train_loss += loss.item()
            output_argmax = output.argmax(1).cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            train_acc = accuracy_score(label, output_argmax)
            train_f1 = f1_score(label, output_argmax, average='micro')

    train_loss /= len(trainloader)
    train_acc /= len(trainloader)

    return train_loss, train_acc, train_f1


def Fusion_test_model(net, testloader, optimizer, device):

   net.eval()
   test_loss = 0
   test_acc = 0

   for x1, x2, label in tqdm(testloader):
       optimizer.zero_grad()
       x1, x2, label = x1.to(device), x2.to(device), label.to(device)

       output = net(x1, x2)
       output = output.repeat(batch_size, 1)

       loss = F.cross_entropy(output.float(), label.long())
       loss.backward(retain_graph=True)

       optimizer.step()
       optimizer.zero_grad()

       with torch.no_grad():
           test_loss += loss.item()
           output_argmax = output.argmax(1).cpu().detach().numpy()
           label = label.cpu().detach().numpy()


           test_acc = accuracy_score(label, output_argmax)
           test_f1 = f1_score(label, output_argmax, average='micro')


   test_loss /= len(testloader)
   test_acc /=len(testloader)

   return  test_loss, test_acc, test_f1



#Execute
net = newnet
lr = 0.0001

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
criterion = nn.MSELoss()
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

Epoch = 500


for epoch in range(Epoch):
    print('epoch:', epoch)
    train_loss, train_acc, train_f1, train_time = Fusion_train_model(net, train_loader, optimizer, device)
    print('train loss ', train_loss, 'train_acc ', train_acc, 'train_f1', train_f1)

    test_loss, test_acc, test_f1, test_time = Fusion_test_model(net, test_loader, optimizer, device)
    print('test loss ', test_loss, 'test_acc ', test_acc, 'test_f1', test_f1)