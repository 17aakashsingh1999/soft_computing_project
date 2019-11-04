import os

split = 0.85


files = os.listdir('processed/benign')

for i in range(len(files)):
    if i < len(files) * split:
        os.system(f'cp processed/benign/{files[i]} dataset/train/benign/{files[i]}')
    else:
        os.system(f'cp processed/benign/{files[i]} dataset/test/benign/{files[i]}')

files = os.listdir('processed/malignant')

for i in range(len(files)):
    if i < len(files) * split:
        os.system(f'cp processed/malignant/{files[i]} dataset/train/malignant/{files[i]}')
    else:
        os.system(f'cp processed/malignant/{files[i]} dataset/test/malignant/{files[i]}')

