# models.py

import torch.nn as nn
import torch.nn.functional as F

class CNN_CBISDDSM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_CBISDDSM, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # Assuming input images resized to 224x224
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 56, 56]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
