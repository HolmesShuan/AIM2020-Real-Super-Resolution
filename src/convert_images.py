import os
import random
import shutil
from PIL import Image
import PIL

for root, dirs, files in os.walk('../TestLRX2/TestLR'):
    for name in files:
        if name[-3:] == 'png':
            in_path = os.path.join(root, name)
            img = Image.open(in_path)
            img = img.convert('RGB')
            img.save(in_path)