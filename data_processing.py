import torch
import torchvision
import os
from typing import Union
import torchvision.transforms as transforms
from utils import Flatten, OneHot, DataToTensor

__all__ = ["get_train_set_by_cls", "get_test_set_by_cls"]


def get_data(train=False, test=False, flatten=False, one_hot=False):
    list_trans = [transforms.RandomHorizontalFlip(),
                  transforms.RandomCrop(32, 4),
                  transforms.Grayscale(),
                  transforms.ToTensor()]
    list_target_trans = [DataToTensor()]
    if flatten:
        list_trans.append(Flatten())
    if one_hot:
        list_target_trans.append(OneHot(10, to_float=True))

    trans = transforms.Compose(list_trans)
    target_trans = transforms.Compose(list_target_trans)

    if train:
        ts = torchvision.datasets.CIFAR10(root="datasets", train=True, download=True, transform=trans,
                                          target_transform=target_trans)
        return ts
    if test:
        list_test_trans = [transforms.Grayscale(), transforms.ToTensor()]
        if flatten:
            list_test_trans.append(Flatten())
        ts = torchvision.datasets.CIFAR10(root="datasets", train=False, download=True,
                                          transform=transforms.Compose(list_test_trans),
                                          target_transform=target_trans)
        return ts


def get_dataset_by_cls(cls: Union[list, int], train=False, test=False, flatten=False, one_hot=False):
    assert train + test == 1
    if isinstance(cls, int):
        cls = [cls]
    data = get_data(train=train, test=test, flatten=flatten, one_hot=one_hot)
    parent_path = os.path.dirname(__file__)
    ind_path = os.path.join(parent_path, "indices", "cls_{}_{}_indices.pt".format(cls, "train" if train else "test"))

    file_exists = os.path.exists(ind_path)
    if file_exists:
        print("Loading indices...")
        indices = torch.load(ind_path)
    else:
        print("Extracting indices...")
        counter = 0
        indices = []
        for inp, lab in data:
            if lab.item() in cls:
                indices.append(counter)
            counter += 1

        if not os.path.exists(os.path.join(parent_path, "indices")):
            os.mkdir(os.path.join(parent_path, "indices"))
        torch.save(indices, ind_path)
        print("Indices saved to {}".format(ind_path))
    dataset = torch.utils.data.Subset(data, indices)

    return dataset


def get_train_set_by_cls(cls: Union[list, int], flatten=False, one_hot=False):
    """
    :param cls: extract classes in cls (or single class if provided with int value); e.g. cls = [0, 1, 8] will extract
    airplane, automobile and ship images
    :param flatten: flatten the images from to a size like (N, features), where N is the batch size
    :param one_hot: one-hot the labels from size of (N) to (N, N_CLASSES), where N is the batch size. In the second
    dimension, only the corresponding position is 1 and other positions are 0.
    :return: a dataset (https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) which is a subset of
    CIFAR10 that contains desired images
    """
    return get_dataset_by_cls(cls, train=True, flatten=flatten, one_hot=one_hot)


def get_test_set_by_cls(cls: Union[list, int], flatten=False, one_hot=False):
    """
    Similar, but for test images.
    """
    return get_dataset_by_cls(cls, test=True, flatten=flatten, one_hot=one_hot)
