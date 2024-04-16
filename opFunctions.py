import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pathlib
from PIL import Image
from typing import Tuple
import os
from tqdm.auto import tqdm


def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              lossFunction: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler,
              device=None):
  model.train()

  trainLoss, trainAcc = 0,0 

  for batch, (X, y) in enumerate(dataLoader):
    X, y = X.to(device), y.to(device)
    
    yPred = model(X)

    loss = lossFunction(yPred, y)
    trainLoss += loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    yPredClass = torch.argmax(torch.softmax(yPred, dim=1), dim=1)
    trainAcc += (yPredClass==y).sum().item()/len(yPred)*100

  scheduler.step()
  lr = optimizer.param_groups[0]['lr']
  trainLoss = trainLoss/ len(dataLoader)
  trainAcc = trainAcc/ len(dataLoader)

  return trainLoss, trainAcc, lr 

def testStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              lossFunction: torch.nn.Module,
              device=None):
  model.eval()

  testLoss, testAcc = 0,0

  with torch.inference_mode():
    for batchh, (X,y) in enumerate(dataLoader):
      X, y = X.to(device), y.to(device)

      testPred = model(X)

      loss = lossFunction(testPred, y)
      testLoss += loss.item()

      testPredLabels = torch.argmax(torch.softmax(testPred, 1), 1)
      testAcc += (testPredLabels==y).sum().item()/len(testPred)*100
  
  testLoss = testLoss / len(dataLoader)
  testAcc = testAcc / len(dataLoader)

  return testLoss, testAcc

def train(model: torch.nn.Module,
          trainDataLoader: torch.utils.data.DataLoader,
          testDataLoader: torch.utils.data.DataLoader,
          lossFunction: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          epochs: int = 5,
          device=None):
  
  result = {"TrainLoss": [],
            "TrainAcc": [],
            "LearningRate": [],
            "TestLoss": [],
            "TestAcc": []}

  for epoch in tqdm(range(epochs)):
    Loss, Acc, lr = trainStep(model, trainDataLoader, lossFunction, optimizer, scheduler, device)

    testLoss, testAcc = testStep(model, testDataLoader, lossFunction, device)

    print(f"Epoch: {epoch} | Loss: {Loss:.4f} Accuracy: {Acc:.2f} %  Learning Rate: {lr} | Test loss: {testLoss:.4f} Test Accuracy: {testAcc:.2f} %")

    result['TrainLoss'].append(Loss)
    result['TrainAcc'].append(Acc)
    result['LearningRate'].append(lr)
    result['TestLoss'].append(testLoss)
    result['TestAcc'].append(testAcc)
  
  return result

def findClasses(dirPath):
    classes = [d for d in os.listdir(dirPath) if os.path.isdir(os.path.join(dirPath, d))]
    classes.sort()
    classToIdx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, classToIdx

class setupDataset(Dataset):
    def __init__(self, targDir: str, mode: str = 'train', transform=None):
        assert mode in ['train', 'test'], "Mode must be 'train' or 'test'"
        self.transform = transform
        self.classes, self.classToIdx = findClasses(targDir)
        
        # Determine sample size based on mode
        numSamplesTrain = 80 
        numSampleTest = 20
        
        # Collect images
        self.paths = []
        for className in self.classes:
            classPath = pathlib.Path(targDir) / className
            allImages = []
            for sequenceDir in classPath.iterdir():
                if sequenceDir.is_dir():
                    allImages.extend(sequenceDir.glob('*.jpg'))
            
            # Ensure list conversion for random sampling and limit per class
            allImages = list(allImages)
            sampledImages = allImages[:numSamplesTrain] if mode == "train" else allImages[numSamplesTrain:numSamplesTrain+numSampleTest]
            self.paths.extend(sampledImages)
  
    def loadImage(self, index: int) -> Image.Image:
        return Image.open(self.paths[index])
  
    def __len__(self) -> int:
        return len(self.paths)
  
    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        img = self.loadImage(index)
        className = self.paths[index].parent.parent.name
        classIdx = self.classToIdx[className]
        if self.transform:
            img = self.transform(img)
        return img, classIdx