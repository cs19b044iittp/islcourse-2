import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score,recall_score, precision_recall_fscore_support
from torchmetrics import Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import accuracy


def load_data():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, test_data

def create_dataloaders(training_data, test_data, batch_size=64):

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
        
    return train_dataloader, test_dataloader


class cs19b003(nn.Module):
    def __init__(self, config_param):
        super().__init__()
        self.config_param = config_param
        
        self.conv1 = nn.Conv2d(in_channels=config_param[1][0], 
                               out_channels=config_param[1][1], 
                               kernel_size=config_param[1][2], 
                               stride=config_param[1][3], 
                               padding=config_param[1][4])
        self.conv2 = nn.Conv2d(in_channels=config_param[2][0], 
                               out_channels=config_param[2][1], 
                               kernel_size=config_param[2][2], 
                               stride=config_param[2][3], 
                               padding=config_param[2][4])
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.fc_nodes_calc(), out_features=config_param[0][4])
        self.m = nn.Softmax(dim =1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.m(x)
        return x

    def fc_nodes_calc(self):
      new_w = self.config_param[0][1]
      new_h = self.config_param[0][2]   
      conv_layers = self.config_param[0][3]
      for i in range(conv_layers):
        # w = (w - k + p + 1)/2
        new_w = (new_w - self.config_param[i+1][2][0] + 1)/self.config_param[i+1][3]
        new_h = (new_h - self.config_param[i+1][2][1] + 1)/self.config_param[i+1][3]
      size = int(self.config_param[conv_layers][1] * new_w * new_h)
      return size

#train the network
def train_network(train_loader, model1, optimizer, criteria, e):
  """
  train_loader = dataloader created for iterating over data
  optimiser = adam optimiser
  criteria = loss fn used for eval
  e = no of epochs
  """
  for epoch in range(e):  # loop over the dataset multiple times
    model1.train()
    model1.to(device)
    running_loss = 0.0
    for i,data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model1(inputs)
        # print(outputs.shape, labels.shape)
        tmp = torch.nn.functional.one_hot(labels, num_classes = 10)
        loss = criteria(outputs, tmp)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 199:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

  print('Finished Training')

#cross entropy
def loss_fun(y_pred, y_ground):
  v = -(y_ground * torch.log(y_pred + 0.0001))
  v = torch.sum(v)
  return v


# testing the model
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            tmp = torch.nn.functional.one_hot(y, num_classes= 10)
            pred = model(X)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #precision_recall_fscore_support(y_ground, y_pred, average='macro')
    accuracy1 = Accuracy()
    accuracy1 = accuracy1.to(device)
    print('Accuracy :', accuracy1(pred,y))
    
    precision = Precision(average = 'macro', num_classes = 10)
    precision = precision.to(device)
    print('precision :', precision(pred,y))

    recall = Recall(average = 'macro', num_classes = 10)
    recall = recall.to(device)
    print('recall :', recall(pred,y))
    
    f1_score = F1Score(average = 'macro', num_classes = 10)
    f1_score = f1_score.to(device)
    print('f1_score :', f1_score(pred,y))

    return accuracy1(pred,y), precision(pred,y), recall(pred,y), f1_score(pred,y)

# test(test_loader, model, loss_fun)

#write the get model
def get_model(train_loader,e,lr,config_param=None):
  model = cs19b003(config_param)
  model=model.to(device)
  
  return model

# test the model in hubconf
def test_model(model1, test_data_loader):
  a,p,r,f1 = test(test_data_loader, model1, loss_fun)
  a=a.to(device)
  p=p.to(device)
  r=r.to(device)
  f1=f1.to(device)
  return a,p,r,f1
