import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch import optim
from torch.backends import cudnn
from torchvision import datasets, transforms, utils as vutils
from torch.autograd import Variable
