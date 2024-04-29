import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision.models import googlenet
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
from torchvision.transforms import RandomCrop, RandomRotation, ColorJitter


aug = 1
def prepare_data(batch_size, mean, std, resize,random_crop_size,valid_split):
    
        # Load and Prepare the CIFAR100 Dataset
    transform = transforms.Compose([
        transforms.Resize(resize),               # Resize to fit GoogLeNet input size
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(15),  # Random rotation between -15 and +15 degrees
        transforms.RandomCrop(random_crop_size, padding=4),  # Pad by 4 pixels and then randomly crop
        transforms.ToTensor(),                # Convert to tensor
        transforms.Normalize((mean,), (std,))  # Normalize with mean and std for 3 channels
    ])


    # Load CIFAR-10 datasets
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)


    # Split training set for training and validation
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(valid_split * num_train))

    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=2)
    validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=valid_sampler, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return trainloader, validloader, testloader
        

# Function to evaluate the model
def evaluate_model(loader, model):
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # If the model returns auxiliary outputs, we ignore them during evaluation
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Only use the main output
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
    return 100 * correct / total, total_loss / len(loader)




# Update Hyperparameters and directories as per requirements
batch_size = 128
resize = 256  # Resize to fit GoogLeNet input
random_crop_size = 224
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
valid_split = 0.1
num_epochs = 200
learning_rate = 0.01
momentum = 0.9
# weight_decay=1e-4

# Specify the name or path of the directory to create
dir_name = "results/BS{}-E{}-lrschCosSingle{}-Adam-CE-CIFAR100".format(batch_size,num_epochs,learning_rate)
if aug == 1:
    dir_name = "results/BS{}-E{}-lrschCosSingle{}-Adam-CE-AUG-CIFAR100".format(batch_size,num_epochs,learning_rate)

# Path of the new directory
current_path = os.getcwd()  # Get the current working directory
path = os.path.join(current_path, dir_name)  # Append the new directory to the current path
# Create the directory
if not os.path.exists(path):
    os.makedirs(path)

information = "DATASET: CIFAR100\nMODEL GoogLeNet prebuilt\nTOTAL EPOCHS : {}\nBATCH SIZE : {}\nLearning rate : {}\nLoss : Cross Entropy\nOptimizer: Adam".format(num_epochs,batch_size,  learning_rate)
if aug == 1:
    information = "DATASET: CIFAR100\nData AUGMENTATION+random crop size{} and rotate\nMODEL GoogLeNet prebuilt\nTOTAL EPOCHS : {}\nBATCH SIZE : {}\nLearning rate : {}\nLoss : Cross Entropy\nOptimizer: Adam".format(random_crop_size,num_epochs,batch_size,  learning_rate)

with open(os.path.join(path,'log_train.txt'), 'a') as file:
        file.write(information)
        
        
        
trainloader, validloader, testloader = prepare_data(batch_size = batch_size, 
                                                    mean = 0.1307,
                                                    std = 0.3081, 
                                                    resize = 224, 
                                                    random_crop_size = random_crop_size, 
                                                    valid_split = 0.1)

# Define and adjust the model
model = googlenet(pretrained=False, aux_logits=True, init_weights=True)
model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # Use 3 input channels
model.fc = nn.Linear(1024, 100)  # Adjust final layer for 10 classes

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)


# Training loop
train_losses = []
train_accuracies = []
running_train_losses = []
running_train_accuracies = []
validation_losses = []
validation_accuracies = []


for epoch in range(num_epochs):  # Number of epochs
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # loss = criterion(outputs, labels)
        # Handle potential auxiliary outputs
        if isinstance(outputs, tuple):
            output, aux1, aux2 = outputs
            loss1 = criterion(output, labels)
            loss2 = criterion(aux1, labels)
            loss3 = criterion(aux2, labels)
            loss = loss1 + 0.3 * (loss2 + loss3)  # Scale aux losses as in Inception
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        # If the model returns auxiliary outputs, we ignore them during evaluation
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Only use the main output
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    running_train_losses.append(running_loss/len(trainloader))
    running_train_accuracies.append(100 * correct / total)
    
    model.eval()
    train_accuracy, train_loss = evaluate_model(trainloader, model)
    valid_accuracy, valid_loss = evaluate_model(validloader, model)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    validation_losses.append(valid_loss)
    validation_accuracies.append(valid_accuracy)
    
    log_info = f'\nEpoch {epoch+1}\nTrain mode:\nTrain Loss: {running_train_losses[-1]:.5f}, Train Acc: {running_train_accuracies[-1]:.5f}%\n'
    log_info += f'Evaluate mode:\nTrain Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.5f}%\nVal Loss: {valid_loss:.5f}, Val Acc: {valid_accuracy:.5f}%\n\n'
    
    print(log_info)
    
    
    with open(os.path.join(path,'log_train.txt'), 'a') as file:
        file.write(log_info)
        
        
print('Finished Training')

# Final evaluation on test data
test_accuracy, test_loss = evaluate_model(testloader, model)
log_info = f'Test Loss: {test_loss:.5f}, Test Acc: {test_accuracy:.5f}%'
print(log_info)

with open(os.path.join(path,'log_train.txt'), 'a') as file:
        file.write(log_info)


model_name = "model-GoogLeNet-prebuilt-BS{}-E{}-lrschCosSingle{}-Adam-CE-CIFAR100.pt".format(batch_size,num_epochs,learning_rate)
if aug == 1:
    model_name = "model-GoogLeNet-prebuilt-BS{}-E{}-lrschCosSingle{}-Adam-CE-AUG-CIFAR100.pt".format(batch_size,num_epochs,learning_rate)

torch.save(model, os.path.join(path,model_name))

with open(os.path.join(path,'losses-and-accuracies.txt'), 'a') as file:
    file.write('Running train Losses and accuracies \n')
    file.write(str(running_train_losses)+'\n')
    file.write(str(running_train_losses)+'\n')
    file.write('Losses train valid test\n')
    file.write(str(train_losses)+'\n')
    file.write(str(validation_losses)+'\n')
    file.write(str(test_loss)+'\n')
    file.write('Accuracies train valid test\n')
    file.write(str(train_accuracies)+'\n')
    file.write(str(validation_accuracies)+'\n')
    file.write(str(test_accuracy)+'\n')
        