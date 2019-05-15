from enum import Enum
from glob import glob
from PIL import Image
import numpy as np
import re
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

filenames = glob("UTKFace/*.jpg")
N = len(filenames)
assert N > 0, "Data files not found in UTKFace/*.jpg!"
print(f"Loading {N} UTKFace images")
ages = np.empty(N, dtype=np.uint8)
genders = np.empty(N, dtype=np.bool)
races = np.empty(N, dtype=np.uint8)

class Race(Enum):
    White = 0
    Black = 1
    Asian = 2
    Indian = 3
    Other = 4

def _errorfix(filename):
    """Provide correct info for incorrectly named files"""
    for fix in (
        ("39_1_20170116174525125", 39, 0, 1),
        ("61_1_20170109150557335", 61, 1, 3),
        ("61_1_20170109142408075", 61, 1, 1),
    ):
        if fix[0] in filename: return fix

# Shuffle the files' order
np.random.seed(0)
np.random.shuffle(filenames)

# Read labels from filenames
for i, n in enumerate(filenames):
    m = re.match(r".*/(\d*)_(\d*)_(\d*)_[^/]*", n) or _errorfix(n)
    ages[i], genders[i], races[i] = int(m[1]), int(m[2]), int(m[3])

# Load JPEGs into Numpy array using multiple threads
with ThreadPoolExecutor() as exec:
    images = list(exec.map(lambda n: np.array(Image.open(n)), filenames))

class Torch:
    def __init__(self, device=None):
        self.images = torch.tensor(np.transpose(images, (0, 3, 1, 2)), device=device, dtype=torch.uint8)

    def batch_iter(self, batch_size, image_size=200):
        """Make a random-sampling iterator that outputs images in batches."""
        while True:
            batch = np.random.randint(N - batch_size)
            batch = slice(batch, batch + batch_size)
            real = self.images[batch].to(torch.float32)
            real /= 128.0
            real -= 1.0
            if image_size != real.size(2):
                real = nn.functional.interpolate(real, image_size, mode="area")
            yield real
