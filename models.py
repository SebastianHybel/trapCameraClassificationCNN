import torch
from torch import nn 



class TinyVGGModel(nn.Module):
    def __init__(self,
                 inputLayer: int,
                 hiddenLayer: int,
                 outputLayer: int) -> None:
        super().__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels= inputLayer,
                      out_channels= hiddenLayer,
                      kernel_size= 3,
                      stride= 1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenLayer,
                      out_channels= hiddenLayer,
                      kernel_size= 3,
                      stride= 1,
                      padding= 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=hiddenLayer,
                      out_channels= hiddenLayer,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hiddenLayer,
                      out_channels=hiddenLayer,
                      kernel_size= 3,
                      stride=1, 
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
        self.classiferLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddenLayer*53*53,
                      out_features=outputLayer)
        )
    def forward (self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.classiferLayer(x)
        return x 



class CNNModel(nn.Module):
    def __init__(self,
                 inputLayer: int,
                 outputLayer: int) -> None:
        super().__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels= inputLayer,
                      out_channels= 32,
                      kernel_size= 3,
                      stride= 1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32,
                      out_channels= 32,
                      kernel_size= 3,
                      stride= 1,
                      padding= 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(in_channels= 32,
                      out_channels= 32,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32,
                      out_channels= 32,
                      kernel_size= 3,
                      stride=1, 
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )
         
        self.classiferLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*53*53,
                      out_features=outputLayer),
        )
    def forward (self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.classiferLayer(x)
        return x 
        