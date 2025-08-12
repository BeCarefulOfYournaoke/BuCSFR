import os
import torch.utils.data as data
import numpy as np

from PIL import Image
from torchvision import transforms


class FGVCAircraft(data.Dataset):
    def __init__(
        self,
        root, 
        split="trainval",
        transform=None,
    ):
        self.root = root
        self.split = split
        self.transform = transform

        self.annotation_levels = ["variant", "family", "manufacturer"]

        self.image_files = []
        image_files_ready = False
        self.labels = []

        self.uni_names_of_all_tax_levels = []
        self.class_to_idx = []

        self.num_fine, self.num_coarse = 100, 30

        image_data_folder = os.path.join(self.root, "images")
        for level in self.annotation_levels:
            annotation_file = os.path.join(
                self.root,
                {
                    "variant": "variants.txt",  #100 class
                    "family": "families.txt",   #70 class
                    "manufacturer": "manufacturers.txt",    #30 class
                }[level],
            )
            with open(annotation_file, "r") as f:
                _classes = [line.strip() for line in f]
                self.uni_names_of_all_tax_levels.append(_classes)
            self.class_to_idx.append(
                dict(zip(_classes, range(len(_classes)))),
            )

            _labels_file = os.path.join(self.root, f"images_{level}_{self.split}.txt")
            _labels = []
            with open(_labels_file, "r") as f:
                for line in f:
                    image_name, label_name = line.strip().split(" ", 1)
                    if not image_files_ready:
                        self.image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
                    _labels.append(self.class_to_idx[-1][label_name])
            self.labels.append(_labels)
            image_files_ready = True

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file).convert("RGB")
        # labels = [self.labels[0][idx], self.labels[1][idx], self.labels[2][idx]]

        labels = [self.labels[2][idx], self.labels[0][idx]]

        if self.transform:
            image = self.transform(image)

        return image, idx, *labels

    def get_graph(self):
        M = np.zeros((self.num_fine, self.num_coarse))
        for fine, coarse in zip(self.labels[0], self.labels[2]):
            M[fine, coarse] += 1
        return M


if __name__ == "__main__":
    root = "/media/disk12T/2022-sgh/datasets/fgvc-aircraft-2013b"

    mean_aircraft = [0.541, 0.519, 0.366]
    std_aircraft = [0.275, 0.264, 0.279]
    normalize_tfs_aircraft = transforms.Normalize(mean=mean_aircraft, std=std_aircraft)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize_tfs_aircraft,
            transforms.RandomResizedCrop(224),
        ]
    )

    aircraft_dataset = FGVCAircraft(
        root,
        split="trainval",
        transform=transform,
    )

    print(len(aircraft_dataset))
    for i in range(len(aircraft_dataset.annotation_levels)):
        print(aircraft_dataset.annotation_levels[i], len(aircraft_dataset.uni_names_of_all_tax_levels[i]))
        print(aircraft_dataset.uni_names_of_all_tax_levels[i])

