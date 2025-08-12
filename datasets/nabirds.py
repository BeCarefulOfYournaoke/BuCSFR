import torch
from torch.utils.data import Dataset
import torchvision.transforms.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import os


class MyCustomNabirds(Dataset):
    labels_file = {
        "train": "train_labels_nabirds.npy",
        "test": "test_labels_nabirds.npy",
    }
    split_file = "train_test_split.pickle"

    def __init__(self, root, split="train", transform=None):
        self.labels = np.load(
            os.path.join(root, MyCustomNabirds.labels_file[split])
        ).astype(np.int64)

        with open(os.path.join(root, MyCustomNabirds.split_file), "rb") as f:
            train_paths, test_paths = pickle.load(f)

        self.image_paths = None
        if split == "train":
            self.image_paths = train_paths
        elif split == "test":
            self.image_paths = test_paths
        else:
            raise NotImplementedError("unsupported split file")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index, :]
        # Categories: 555, Species: 495, Genus/Family: 228, and Order: 22

        label = torch.from_numpy(label)
        coarse_label = label[3]
        fine_label = label[0]

        if self.transform is not None:
            image = self.transform(image)

        return image, index, coarse_label, fine_label

    def get_label_stat(self, parent=3, child=2):
        pt = self.labels[:, parent]
        cd = self.labels[:, child]

        stat = []
        for pt_idx in np.unique(pt):
            mask = (pt == pt_idx)
            cd_of_pt_idx = cd[mask]
            stat.append(
                len(np.unique(cd_of_pt_idx))
            )
        return stat


if __name__ == "__main__":
    root = "/media/disk12T/2022-sgh/datasets/NABirds"
    train_dataset = MyCustomNabirds(root=root, split="train")
    stat = train_dataset.get_label_stat(parent=3, child=2)
    print(stat)
