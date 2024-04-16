from typing import List
import torch 
import matplotlib.pyplot as plt
from torch import nn 
import random
import matplotlib.pyplot as plt
import pandas as pd 

def displayRandomImages(dataset: torch.utils.data.Dataset,
                        classes: List[str] = None,
                        n: int = 10,
                        displayShape: bool = True,
                        seed: int = None): 
  # 2. Adjust display if n is too high 
  if n > 10:
    n = 10
    displayShape = False
  
  # 3. Set the seed 
  if seed:
    random.seed(seed)
  
  # 4. Get random sample indexes 
  randomSamplesIdx = random.sample(range(len(dataset)), k=n)

  # 5 setup the plot 
  plt.figure(figsize=(16, 8))

  # 6. Loop through random indexes and plot them with matplotlib 
  for i, targSample in enumerate(randomSamplesIdx):
    targImage, targLabel = dataset[targSample][0], dataset[targSample][1]

    # 7. Adjust tensor dimension for plotting
    targImageAdjust = targImage.permute(1, 2, 0) # CHW -> HWC

    # Plot adjusted samples
    plt.subplot(1, n, i+1)
    plt.imshow(targImageAdjust)
    plt.axis("off")
    if classes:
      title=f"Class: {classes[targLabel]}"
      if displayShape:
        title = title + f"\nshape: {targImageAdjust.shape}"
    plt.title(title)

def plotLossAndAccuracy(csvFilePath: str,
                        iteration: int):
    # Read the CSV file using pandas
  df = pd.read_csv(csvFilePath)
  df['Epoch'] = df['Epoch'].astype(int) 
  
    # Plotting
  fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot Train Loss vs. Test Loss
  axes[0].plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o')
  axes[0].plot(df['Epoch'], df['Test Loss'], label='Test Loss', marker='x')
  axes[0].set_xlabel('Epoch')
  axes[0].set_ylabel('Loss')
  axes[0].set_title('Training vs. Testing Loss')
  axes[0].legend()
    
    # Plot Train Accuracy vs. Test Accuracy
  axes[1].plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy', marker='o')
  axes[1].plot(df['Epoch'], df['Test Accuracy'], label='Test Accuracy', marker='x')
  axes[1].set_xlabel('Epoch')
  axes[1].set_ylabel('Accuracy')
  axes[1].set_title('Training vs. Testing Accuracy')
  axes[1].legend()
    
    # Adjust layout
  plt.tight_layout()

  # Super Title
  plt.suptitle(f'Iteration number {iteration}', fontsize=16, y=1.05)

    # Show plots
  plt.show()