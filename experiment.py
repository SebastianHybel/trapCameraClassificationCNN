from opFunctions import setupDataset, train
from models import TinyVGGModel, CNNModel
import torch
from timeit import default_timer as timer
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os 
import csv

iteration = 3 
EPOCHS = 16
LR = 0.01

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.ColorJitter(brightness=.5, hue=.3),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor()
    
])

testTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

device = "cuda" if torch.cuda.is_available() else "cpu" 

targetDir = "dataset/set"
trainData = setupDataset(targDir= targetDir,
                         mode= "train",
                        transform=transform)

testData = setupDataset(targDir=targetDir,
                        mode="test",
                        transform=testTransform)

classNames = trainData.classes

trainDataLoader = DataLoader(dataset=trainData,
                             batch_size=32,
                             shuffle=True)

testDataLoader = DataLoader(dataset=testData,
                            batch_size=32,
                            shuffle=True)

modelA = CNNModel(inputLayer=3,
                    outputLayer=len(classNames))

lossFunction = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(modelA.parameters(),
                             LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

startTime = timer() 

modelAResults = train(modelA, trainDataLoader, testDataLoader, lossFunction, optimizer, scheduler, EPOCHS, device)

endTime = timer()

os.makedirs("results", exist_ok=True)
os.makedirs("modelsPT", exist_ok=True)

resultsCsvPath = f'results/TrainingResults{iteration}.csv'
modelSavePath = f"modelsPT/ModITER{iteration}.pt"


with open(resultsCsvPath, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])

    num_epochs = len(modelAResults['TrainLoss'])
    
    for i in range(num_epochs):
        writer.writerow([
            i + 1,  
            modelAResults['TrainLoss'][i],
            modelAResults['TrainAcc'][i],
            modelAResults['TestLoss'][i],
            modelAResults['TestAcc'][i]
        ])

torch.save(modelA.state_dict(), modelSavePath)


print(f"Total training time: {endTime-startTime: .3f} seconds ")