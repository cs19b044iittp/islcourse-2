import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def kali():
  print('kali')

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

m1=8
n1=8
pic_len1=625
class Cs19b003NN(nn.Module):
  # pass
  # ... your code ...
  # ... write init and forward functions appropriately ...
    def __init__(self, m, n, pic_len):
        super(Cs19b003NN, self).__init__()
        self.flatten = nn.Flatten()
        self.m = m
        self.n=n
        self.pic_len=pic_len
        print("lens:", m, n, pic_len)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear( pic_len, pic_len),
            nn.ReLU(),
            nn.Linear(pic_len, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
            
        )
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
  # model = None

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  # with torch.no_grad():
  for X, y in train_data_loader:
      X, y = X.to(device), y.to(device)
      print("X and y shape", X.shape, y.shape)
      m=X.shape[0]
      n=X.shape[1]
      pic_len = X.shape[2]*X.shape[3]
      break
  model = Cs19b003NN(m, n, pic_len).to(device)
  
  for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data_loader, model, optimizer)
  
  print ('Returning model... (rollnumber: cs03)', model)
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = None

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  # In addition,
  # Refer to config dict, where learning rate is given, 
  # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
  # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
  # You need to create 2d convoution layers as per specification above in each element
  # You need to add a proper fully connected layer as the last layer
  
  # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
  # HINT: Flatten function can also be used if required
  
  
  print ('Returning model... (rollnumber: cs03)')
  
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
  size = len(test_data_loader.dataset)
  num_batches = len(test_data_loader)
  print("test_data_loader size:", size, " num batch", num_batches)
  
  model1.eval()
  test_loss, correct = 0, 0
  
  with torch.no_grad():
      for X, y in test_data_loader:
          X, y = X.to(device), y.to(device)
          print("X and y shape", X.shape, y.shape)
          
          pred = model1(X)
          test_loss += loss_fn(pred, y).item()
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  print ('Returning metrics... (rollnumber: cs03)')
  
  
  return accuracy_val, precision_val, recall_val, f1score_val
