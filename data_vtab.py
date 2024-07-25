import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.transforms import InterpolationMode
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from torchvision.models import ViT_B_16_Weights


# I think I took this class from somewhere, but I'm not sure
class HFDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.data[idx]

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            label = self.target_transform(label)

        return data, label


def get_dataset_from_HF(name: str, cols: list, transform: transforms = None, target_transform: transforms = None):
    hf_data = load_dataset(name).select_columns(cols)
    split_1k = hf_data["train"].train_test_split(train_size=1000, seed=42)

    hf_train = split_1k["train"]

    if "test" in hf_data:
        hf_test = hf_data["test"]
    else:
        hf_test = split_1k["train"]

    train_set = HFDataset(hf_train['image'], hf_train['label_distance'], transform=transform,
                          target_transform=target_transform)
    test_set = HFDataset(hf_test['image'], hf_test['label_distance'], transform=transform,
                         target_transform=target_transform)

    return train_set, test_set


def download_dataset(name, transform, download=True, seed=42):
    name = name.lower()

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()

    print("Preparing dataset:", name)

    if name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=download, transform=transform)
        trainset, _ = random_split(trainset, [1000, len(trainset) - 1000], generator=generator)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=download, transform=transform)
        labels_count = 100

    elif name == 'caltech101':
        data = torchvision.datasets.Caltech101(root='./data', download=False, transform=transform)
        trainset, testset = random_split(data, [1000, len(data) - 1000], generator=generator)
        labels_count = 102

    elif name == 'dtd':
        trainset = torchvision.datasets.DTD(root='./data', split="train", download=download, transform=transform)
        trainset, _ = random_split(trainset, [1000, len(trainset) - 1000], generator=generator)

        testset = torchvision.datasets.DTD(root='./data', split="test", download=download, transform=transform)
        labels_count = 47

    elif name == 'flowers102':
        trainset = torchvision.datasets.Flowers102(root='./data', split="train", download=download, transform=transform)
        trainset, _ = random_split(trainset, [1000, len(trainset) - 1000], generator=generator)

        testset = torchvision.datasets.Flowers102(root='./data', split="test", download=download, transform=transform)
        labels_count = 102

    elif name == 'pets':
        trainset = torchvision.datasets.OxfordIIITPet(root='./data', split="trainval", download=download,
                                                      transform=transform, target_types="category")
        trainset, _ = random_split(trainset, [1000, len(trainset) - 1000], generator=generator)

        testset = torchvision.datasets.OxfordIIITPet(root='./data', split="test", download=download,
                                                     transform=transform, target_types="category")
        labels_count = 37

    elif name == 'svhn':
        trainset = torchvision.datasets.SVHN(root='./data', split="train", download=download, transform=transform)
        trainset, _ = random_split(trainset, [1000, len(trainset) - 1000], generator=generator)

        testset = torchvision.datasets.SVHN(root='./data', split="test", download=download, transform=transform)
        labels_count = 10

    elif name == 'sun397':

        data = torchvision.datasets.SUN397(root='./data', download=download, transform=transform)
        trainset, testset = random_split(data, [1000, len(data) - 1000], generator=generator)

        labels_count = 397

    elif name == 'pcam':
        trainset = torchvision.datasets.PCAM(root='./data', split="train", download=download, transform=transform)
        trainset, _ = random_split(trainset, [1000, len(trainset) - 1000], generator=generator)

        testset = torchvision.datasets.PCAM(root='./data', split="test", download=download, transform=transform)
        labels_count = 2

    elif name == 'eurosat':
        data = torchvision.datasets.EuroSAT(root='./data', download=download, transform=transform)
        trainset, testset = random_split(data, [1000, len(data) - 1000], generator=generator)

        labels_count = 18


    elif name == 'resisc45':
        # trainset = RESISC45(root='./data', split="train", download=download, transforms=transform)
        # trainset, _ = random_split(trainset, [1000, len(trainset) - 1000], generator=generator)
        #
        # testset = RESISC45(root='./data', split="test", download=download, transforms=transform)

        hf_train = load_dataset("dpdl-benchmark/resisc45", split="train")
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train, hf_test = splitted["train"], splitted["test"]

        trainset = HFDataset(hf_train['image'], hf_train['label'], transform)
        testset = HFDataset(hf_test['image'], hf_test['label'], transform)

        labels_count = 45


    elif name == 'kitti':
        # trainset = torchvision.datasets.Kitti(root='./data', train=True, download=download, transform=transform,
        #                                       target_transform=transforms.ToTensor())
        # trainset, _ = random_split(trainset, [1000, len(trainset) - 1000], generator=generator)
        #
        # testset = torchvision.datasets.Kitti(root='./data', train=False, download=download, transform=transform,
        #                                      target_transform=transforms.ToTensor())

        hf_train = load_dataset("dpdl-benchmark/kitti", split="train").select_columns(['image', 'label_distance'])
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train = splitted["train"]

        hf_test = load_dataset("dpdl-benchmark/kitti", split="test").select_columns(['image', 'label_distance'])

        trainset = HFDataset(hf_train['image'], hf_train['label_distance'], transform)
        testset = HFDataset(hf_test['image'], hf_test['label_distance'], transform)

        labels_count = 4


    elif name == 'smallnorb_az':
        hf_train = load_dataset("dpdl-benchmark/smallnorb", split="train").select_columns(['image', 'label_azimuth'])
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train = splitted["train"]

        hf_test = load_dataset("dpdl-benchmark/smallnorb", split="test").select_columns(['image', 'label_azimuth'])

        trainset = HFDataset(hf_train['image'], hf_train['label_azimuth'], transform)
        testset = HFDataset(hf_test['image'], hf_test['label_azimuth'], transform)

        labels_count = 18

    elif name == 'smallnorb_el':
        hf_train = load_dataset("dpdl-benchmark/smallnorb", split="train").select_columns(['image', 'label_elevation'])
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train = splitted["train"]

        hf_test = load_dataset("dpdl-benchmark/smallnorb", split="test").select_columns(['image', 'label_elevation'])

        trainset = HFDataset(hf_train['image'], hf_train['label_elevation'], transform)
        testset = HFDataset(hf_test['image'], hf_test['label_elevation'], transform)

        labels_count = 9

    elif name == 'clevr_count':
        hf_train = load_dataset("dpdl-benchmark/clevr", split="train").select_columns(['image', 'label_count'])
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train = splitted["train"]

        hf_test = load_dataset("dpdl-benchmark/clevr", split="test").select_columns(['image', 'label_count'])

        trainset = HFDataset(hf_train['image'], hf_train['label_count'], transform)
        testset = HFDataset(hf_test['image'], hf_test['label_count'], transform)

        labels_count = 8

    elif name == 'clevr_dist':
        hf_train = load_dataset("dpdl-benchmark/clevr", split="train").select_columns(['image', 'label_distance'])
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train = splitted["train"]

        hf_test = load_dataset("dpdl-benchmark/clevr", split="test").select_columns(['image', 'label_distance'])

        trainset = HFDataset(hf_train['image'], hf_train['label_distance'], transform)
        testset = HFDataset(hf_test['image'], hf_test['label_distance'], transform)

        labels_count = 6

    elif name == 'dmlab':
        hf_train = load_dataset("dpdl-benchmark/dmlab", split="train")
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train = splitted["train"]

        hf_test = load_dataset("dpdl-benchmark/dmlab", split="test")

        trainset = HFDataset(hf_train['image'], hf_train['label'], transform)
        testset = HFDataset(hf_test['image'], hf_test['label'], transform)

        labels_count = 6

    elif name == 'dsprites_or':
        hf_train = load_dataset("dpdl-benchmark/dsprites", split="train")
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train, hf_test = splitted["train"], splitted["test"]

        trainset = HFDataset(hf_train['image'], hf_train['label_orientation'], transform)
        testset = HFDataset(hf_test['image'], hf_test['label_orientation'], transform)

        labels_count = 40

    elif name == 'dsprites_loc':
        hf_train = load_dataset("dpdl-benchmark/dsprites", split="train")
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train, hf_test = splitted["train"], splitted["test"]

        trainset = HFDataset(hf_train['image'], hf_train['label_x_position'], transform)
        testset = HFDataset(hf_test['image'], hf_test['label_x_position'], transform)

        labels_count = 32


    elif name == 'retinopathy':
        hf_train = load_dataset("NawinCom/Eye_diabetic", split="train").select_columns(['image', 'labels'])
        splitted = hf_train.train_test_split(train_size=1000, seed=42)
        hf_train = splitted["train"]

        hf_test = load_dataset("NawinCom/Eye_diabetic", split="validation").select_columns(['image', 'labels'])

        trainset = HFDataset(hf_train['image'], hf_train['labels'], transform)
        testset = HFDataset(hf_test['image'], hf_test['labels'], transform)

        labels_count = 5


    else:
        raise ValueError('Wrong dataset name.')

    return trainset, testset, labels_count


def is_greyscale(dataset_name):
    return dataset_name.lower() in ['smallnorb_az', 'smallnorb_el', 'dsprites_or', 'dsprites_loc']


def test_dataset(name):
    if is_greyscale(name):
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    trainset, testset, labels_count = download_dataset(name, transform, True)

    print(trainset[0])

    print(len(trainset), len(testset), labels_count)

    loader = DataLoader(trainset, batch_size=1, shuffle=False)
    x, y = next(iter(loader))

    print(x.shape, y.shape)


if __name__ == '__main__':
    test_dataset('kitti')
