import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# basic checks
print(torch.__version__)
print(torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())



