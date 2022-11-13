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
!pip install torchmetrics
from torchmetrics import Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import accuracy

transform_tensor_to_pil = ToPILImage()
transform_pil_to_tensor = ToTensor()

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

training_data, test_data = load_data()
def create_dataloaders(training_data, test_data, batch_size=64):

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
        
    return train_dataloader, test_dataloader

# configs 
num_classes = len(set([y for x,y in training_data]))
dim, width, height = training_data[0][0].shape
conv_layers=2
config_param=[[dim, width, height, conv_layers, num_classes],
              [dim, 32, (5,5), 1, 'valid'],
              [ 32, 64, (5,5), 1, 'valid']]

print ("num classes:", num_classes)
print("config: ", config_param)

train_loader, test_loader = create_dataloaders(training_data, test_data, batch_size = 32)

class cs19b003(nn.Module):
    def __init__(self, config_param):
        super().__init__()
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
        self.fc1 = nn.Linear(in_features=self.fc_nodes_calc(), out_features=num_classes)
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
      new_w = config_param[0][1]
      new_h = config_param[0][2]
      for i in range(conv_layers):
        # w = (w - k + p + 1)/2
        new_w = (new_w - config_param[i+1][2][0] + 1)/config_param[i+1][3]
        new_h = (new_h - config_param[i+1][2][1] + 1)/config_param[i+1][3]
      size = int(config_param[conv_layers][1] * new_w * new_h)
      return size

# y = (len(set([y for x,y in training_data])))
model = cs19b003(config_param)
model = model.to(device)
from torchsummary import summary
print("model", model)

summary(model, (1,28,28))
# ouputs <batch_size, num_channels, width, height>
# out num_channels of a layer = in num_channels of filter
# FC input = num_channels x width x height

# https://towardsdatascience.com/understanding-input-and-output-shapes-in-convolution-network-keras-f143923d56ca

#train the network
def train_network(train_loader, optimizer, criteria, e):
  """
  train_loader = dataloader created for iterating over data
  optimiser = adam optimiser
  criteria = loss fn used for eval
  e = no of epochs
  """
  for epoch in range(e):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for i,data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
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

# Test for single datapoint
x,y = training_data[0]
model = cs19b003(config_param)
model = model.to(device)
tensor=torch.rand((1,1,28,28))
tensor[0]=x
y_pred = model(tensor.to(device))
print("y_pred.shape", y_pred.shape)
print("y_pred", y_pred)
print("sum y_pred", torch.sum(y_pred))
#cross_entropy(10,y_pred)

y_ground = y
loss_val = loss_fun(y_pred, y_ground)
print(loss_val)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_network(train_loader,optimizer,loss_fun,10)

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
    return accuracy1,precision, recall, f1_score

test(test_loader, model, loss_fun)
#write the get model
def get_model(train_loader,e = 10):
	model = cs19b003(config_param)
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	criteria = loss_fun
	train_network(train_loader, optimizer,criteria,e)
	return model
