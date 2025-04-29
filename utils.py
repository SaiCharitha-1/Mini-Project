import copy
import torch
from torchvision import datasets, transforms

from dataclasses import dataclass
from models import CNN_CBISDDSM
from datasets import CBISDDSM_Dataset
from sampling import cbisddsm_iid  # Add this import

@dataclass
class ARGS:
    model: str = "cnn"   # Fixed: now model is CNN
    optimizer: str = 'adam'
    num_channels: int = 3  # CBIS-DDSM images are RGB (not grayscale)
    kernel_num: int = 9
    kernel_size: tuple = (3, 4, 5)
    norm: str = "batch_norm"
    num_filters: int = 32
    max_pool: bool = True
    lr: float = 1e-3
    epochs: int = 10
    iid: bool = True
    frac: float = 0.1
    num_users: int = 10
    local_bs: int = 32
    local_ep: int = 5
    dataset: str = 'cbis_ddsm'  # Fixed: default dataset
    num_classes: int = 2
    gpu: bool = False
    verbose: bool = False
    momentum: float = 0.9
    unequal: int = 0 
    seed: int = 1

def get_dataset(args):
    """ Returns train and test datasets and user group (optional). """

    if args.dataset == 'cbis_ddsm':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_labels_file = '../data/cbis_ddsm/train_labels.csv'
        test_labels_file = '../data/cbis_ddsm/test_labels.csv'

        train_dataset = CBISDDSM_Dataset(image_dir='../data/cbis_ddsm/train', label_file=train_labels_file, transform=transform)
        test_dataset = CBISDDSM_Dataset(image_dir='../data/cbis_ddsm/test', label_file=test_labels_file, transform=transform)
        
        user_groups = None

        return train_dataset, test_dataset, user_groups

    else:
        exit('Error: unrecognized dataset')


def average_weights(w):
    """Returns the average of the weights."""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

@torch.no_grad()
def calc_accuracy(model, criterion, test_loader, device):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    loss_avg = loss / len(test_loader)
    accuracy = correct / total
    return accuracy, loss_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
