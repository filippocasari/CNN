 # @author: Filippo Casari

import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torchvision.transforms as transforms
from numpy import asarray
from numpy import savetxt
torch.manual_seed(17)
print("ciao")

def show(data):
  fig, axes=plt.subplots(nrows=4, ncols=4)
  count=0
  for i in axes:
    for j in i:
      j.imshow(data.data[count])
      count+=1

#plt.imshow(train_set.data[2])

#plt.imshow(train_set.data[0])

device = torch.device(
"cuda:0" if torch.cuda.is_available() else "cpu")
# Datasets
train_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root='./data',train=False, transform=transforms.ToTensor())
# Dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_set,  shuffle=True)
test_loader = torch.utils.data.DataLoader(
dataset=test_set, shuffle=False)
print("number of images: ", len(train_set.data))

images=[img[0] for img in train_set]
images=torch.stack(images, dim=0)
mean=np.array([torch.mean(images[:,0,:,:]), torch.mean(images[:,1,:,:]), torch.mean(images[:,2,:,:])])
std=np.array([torch.std(images[:,0,:,:]), torch.std(images[:,1,:,:]), torch.std(images[:,2,:,:])])
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
imgs_train_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=True, transform=transform_norm)
imgs_test_set=torchvision.datasets.CIFAR10(root='./data',train=False,download=True, transform=transform_norm)

idx = np.arange(len(train_set))
val_indices = idx[50000-1000:]
train_indices= idx[0:49000]
tr1 = torch.utils.data.SubsetRandomSampler(train_indices)
tr2 = torch.utils.data.SubsetRandomSampler(val_indices)
trainset_ = torch.utils.data.DataLoader(train_set, sampler=tr1, batch_size=32 )
val_set =torch.utils.data.DataLoader(train_set, sampler=tr2, batch_size=32 )
print(len(trainset_))
print(len(val_set))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.20) # adding drop out, first dropout=0.15
        # -----2 convolutional layers + max polling -------
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.30)
        # first drop out= 0.20
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.dropout3 = nn.Dropout(0.50)
        self.fc2=nn.Linear(512, 10)
        

    def forward(self, x):
        x = (F.relu(self.conv1(x))) 
        x = F.relu(self.conv2(x)) 
        x=self.pool1(x)
        x=self.dropout1(x)
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 
        x = self.pool2(x) 
        x=self.dropout2(x)
        x = x.view(-1, 64 * 5 * 5) 
        x = F.relu(self.fc1(x)) 
        x=self.dropout3(x)
        x= self.fc2(x)
        return x

new_model=Net()
new_model.to(device)
print("current device used: ", device)
print(new_model)
lr=10**(-3)
momentum=0.9
batch_size=32
epochs=45
model_optimizer=torch.optim.SGD(params=new_model.parameters(), lr=lr, momentum=momentum)
loss_F=nn.CrossEntropyLoss()

TrainLoss=[]
TrainAcc=[]
ValLoss=[]
ValAcc=[]
#savetxt("ciao.csv", ValLoss, delimiter=',')
print("Number of device CUDA: ",torch.cuda.device_count() )
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  new_model = nn.DataParallel(new_model)

new_model.to(device)
 # first contains best validation accuracy while the second element cointains the best epoch  
best_values=[0]*2

for epoch in range(1, epochs):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    run_step = 0
    new_model.train()
    for i, (images, labels) in enumerate(trainset_,0):
        
        # shape of input images is (B, 1, 28, 28).
        #images = images.view(-1, 32*32)  # reshape to (B, 784).
        images = images.to(device)
        labels = labels.to(device)  # shape (B).
        outputs = new_model(images)  # shape (B, 10).
        loss = loss_F(outputs, labels)
        model_optimizer.zero_grad()  # reset gradients.
        loss.backward()  # compute gradients.
        model_optimizer.step()  # update parameters.

        running_loss += loss.item()
        running_total += labels.size(0)
        running_acc_last_value = (100 * running_correct / running_total)
        with torch.no_grad():
            _, predicted = outputs.max(1)
        running_correct += (predicted == labels).sum().item()
        run_step += 1
        if i % 200 == 0:
            # check accuracy.
            print(f'epoch: {epoch}, steps: {i}, '
                  f'train_loss: {running_loss / run_step :.3f}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
            #running_acc_last_value = (100 * running_correct / running_total)
            #running_loss = 0.0
            #running_total = 0
            #running_correct = 0
            #run_step = 0
    TrainLoss.append(running_loss / run_step)
    TrainAcc.append(running_acc_last_value)
    print(f'epoch: {epoch}, '
                  f'train_loss: {running_loss / run_step :.3f}, '
                  f'running_acc: {100 * running_correct / running_total:.1f} %')
    # validate
    correct = 0
    total = 0
    new_model.eval()
    run_step=0
    running_loss=0
    with torch.no_grad():

        for data in val_set:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = new_model(images)
            loss = loss_F(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            run_step+=1
    val_acc = 100 * correct / total

            
    if(best_values[0]<val_acc):
                best_values[0]=val_acc
                best_values[1]=epoch 
    print(f'Validation accuracy: {100 * correct / total} %')
    print(f'Validation error rate: {100 - 100 * correct / total: .2f} %')
    ValLoss.append((running_loss/run_step))
    print(f'Validation Loss: {running_loss/run_step}')
    ValAcc.append((100 * correct / total))
print('Finished Training')
savetxt('ValLoss.csv', ValLoss, delimiter=',')
savetxt('TrainLoss.csv', ValLoss, delimiter=',')
savetxt('ValAcc.csv', ValLoss, delimiter=',')
savetxt('TrainAcc.csv', ValLoss, delimiter=',')

plt.figure(figsize=(12, 12))
plt.plot(np.arange(0, epochs-1), ValLoss, color="red", label="VALIDATION LOSS")

plt.plot(np.arange(0, epochs-1), TrainLoss, color="green", label="TRAINING LOSS")
print(ValLoss)
min_validation_point=min(ValLoss)
x_min_val_value=ValLoss.index(min_validation_point)
plt.scatter(x_min_val_value,min_validation_point, color="black", label="MINIMUM OF VAL LOSS")
plt.legend()
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(np.arange(0, epochs-1), ValAcc, color="red", label="Validation Accuracy")
plt.plot(np.arange(0, epochs-1), TrainAcc, color="green", label="Training Accuracy")
max_acc=max(ValAcc)
x_max_acc=ValAcc.index(max_acc)
plt.scatter(x_max_acc, max_acc, color="black", label="Maximum Validation Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

# Evaluation Test Set

with torch.no_grad():
  correct = 0 
  total = 0
  new_model.eval() # Set model in eval mode. 
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = new_model(images) 
    _, predicted = outputs.max(dim=1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
print(f'Test accuracy is: {test_acc} %')
