from enum import Enum
from glob import glob
from PIL import Image
import numpy as np
import re

filenames = glob("UTKFace/*.jpg")
N = len(filenames)
images = np.empty((N, 200, 200, 3), dtype=np.uint8)
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


for i, n in enumerate(filenames):
    m = re.match(r".*/(\d*)_(\d*)_(\d*)_[^/]*", n) or _errorfix(n)
    ages[i], genders[i], races[i] = int(m[1]), int(m[2]), int(m[3])
    images[i] = np.array(Image.open(n))
