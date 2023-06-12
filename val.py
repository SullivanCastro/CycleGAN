import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage

import numpy as np
from PIL import Image

from parameters import BATCH_SIZE, N_WORKERS, DATASET_PATH, IMG_SIZE
from dataset import Dataset
from model import CycleGAN

import matplotlib.pyplot as plt

# Load the model
cycleGan = CycleGAN(G_X2Y=True, G_Y2X=True, D_X=True, D_Y=True)

# Eval the model
cycleGan.val()

