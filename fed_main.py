# main.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import CBISDDSM_Dataset
from sampling import cbisddsm_iid
from models import CNN_CBISDDSM
from update import LocalUpdate, test_inference

# Arguments (example, usually you have an Args class)
class Args:
    local_bs = 32
    local_ep = 5
    lr = 0.001
    gpu = True
    optimizer = 'adam'
    verbose = 1

args = Args()

# Prepare dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CBISDDSM_Dataset(image_dir='./CBIS_DDSM/images', label_file='./CBIS_DDSM/labels.csv', transform=transform)
num_users = 10
dict_users = cbisddsm_iid(dataset, num_users)

# Initialize model
model = CNN_CBISDDSM(num_classes=2)
model.to('cuda' if args.gpu else 'cpu')

# Example: train local model for user 0
local = LocalUpdate(args=args, dataset=dataset, idxs=dict_users[0], logger=None)
w, loss = local.update_weights(model=model, global_round=1)

# Test global model
acc, test_loss = test_inference(args, model, dataset)
print(f"Test Accuracy: {acc*100:.2f}%, Test Loss: {test_loss:.4f}")
