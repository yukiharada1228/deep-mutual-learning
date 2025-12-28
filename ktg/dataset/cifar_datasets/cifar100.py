# Import packages
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

from ..custom_dataset import CustomDataset


def _compute_mean_std(subset_like):
    loader = DataLoader(
        CustomDataset(subset_like, transform=transforms.ToTensor()), batch_size=64
    )
    h, w = 0, 0
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.cuda()
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
    mean = chsum / len(subset_like) / h / w
    chsum = None
    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.cuda()
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(chsum / (len(subset_like) * h * w - 1))
    mean = mean.view(-1).cpu().numpy()
    std = std.view(-1).cpu().numpy()
    return mean, std


def get_datasets(use_test_mode: bool = False):
    dataset = datasets.CIFAR100(root="data", train=True, download=True)
    lengths = [40000, 10000]
    subsets = random_split(dataset, lengths)
    train_subset, val_subset = subsets[0], subsets[1]

    # Normalization statistics:
    # - Normal: train/validation use train statistics
    # - Test mode (use_test_mode=True): train(=train+val) and test use train+val statistics
    mean_train, std_train = _compute_mean_std(train_subset)
    mean_all, std_all = _compute_mean_std(ConcatDataset([train_subset, val_subset]))

    print("train mean: %s" % mean_train)
    print("train std : %s" % std_train)
    print("test  mean (train+val): %s" % mean_all)
    print("test  std  (train+val): %s" % std_all)

    if use_test_mode:
        train_all = ConcatDataset([train_subset, val_subset])
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean_all, std_all),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean_all, std_all),
            ]
        )
        train_dataset = CustomDataset(train_all, transform=train_transform)
        test_dataset = datasets.CIFAR100(
            root="data", train=False, download=True, transform=test_transform
        )
        # When use_test_mode=True, return only (train, test)
        return train_dataset, test_dataset
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean_train, std_train),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean_train, std_train),
            ]
        )
        train_dataset = CustomDataset(train_subset, transform=train_transform)
        val_dataset = CustomDataset(val_subset, transform=val_transform)
        # Normally return only (train, val)
        return train_dataset, val_dataset
