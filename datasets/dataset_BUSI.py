import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
import itertools
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from PIL import ImageFilter
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample

# 将PNG图像和标签转换为H5文件
def convert_png_to_h5(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    for img_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files)):
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, label_file)

        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')

        image_np = np.array(image).astype(np.float32) / 255.0
        label_np = np.array(label).astype(np.float32)

        case_name = os.path.splitext(img_file)[0]
        h5_path = os.path.join(save_dir, f'{case_name}.h5')
        with h5py.File(h5_path, 'w') as h5f:
            h5f.create_dataset('image', data=image_np)
            h5f.create_dataset('label', data=label_np)

class BUSI_dataset(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split == "train":
            list_file = os.path.join(self._base_dir, "train_slices.list")
        elif self.split == "val":
            list_file = os.path.join(self._base_dir, "val.list")

        with open(list_file, "r") as f:
            self.sample_list = f.readlines()
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(os.path.join(self._base_dir, "data", f"{case}.h5"), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}

        if self.split == "train":
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class BUSI_dataset_aug(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split == "train":
            list_file = os.path.join(self._base_dir, "train_slices.list")
        elif self.split == "val":
            list_file = os.path.join(self._base_dir, "val.list")

        with open(list_file, "r") as f:
            self.sample_list = f.readlines()
        self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(os.path.join(self._base_dir, "data", f"{case}.h5"), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]

        img = Image.fromarray((image * 255).astype(np.uint8))
        img_s1 = deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        image = np.array(img_s1).astype(image.dtype) / 255.0

        sample = {"image": image, "label": label}

        if self.split == "train":
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)