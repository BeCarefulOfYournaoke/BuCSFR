# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from glob import glob
from PIL import Image
from operator import itemgetter

from icecream import ic


#  iNaturalist 2019 dataset
SPLIT_FILE = {
    "train": "train_split",
    "val": "val_split",
    "trainval": "trainval_split",
    "test": "test_split",
}

# IMAGE_FILE = {
#     "train": "train_image",
#     "val": "val_image",
#     "trainval": "trainval_image",
#     "test": "test_image",
# }

IMAGE_FILE = {
    "train": "train_image_resized",
    "val": "val_image_resized",
    "trainval": "trainval_image_resized",
    "test": "test_image_resized",
}


# Category names at each level
CATE_FILE = "categories.json"

# Name mapping at each levels
TAX_LEVELS_MAPPING = {
    "species": 0,  
    "genus": 1,  
    "family": 2,  
    "order": 3,  
    "class": 4,  
    "phylum": 5,  
    "kingdom": 6,  
}

# The number of categories at each level
TAXONOMY_NUM = {
    "species": 1010,
    "genus": 72,
    "family": 57,
    "order": 34,
    "class": 9,
    "phylum": 4,
    "kingdom": 3,
}


class iNaturalist_2019(data.Dataset):
    def __init__(self, root, mode="train", transform=None):
        self.mode = mode

        # attain the category names at each level
        cate_path = os.path.join(root, CATE_FILE)
        cate_data_dict = self._read_categories(cate_path)

        # Obtain the non-repetitive category names at each level and the category labels at each level
        self.uni_names_of_all_tax_levels = []
        self.class_labels_of_all_tax_levels = []
        self._get_tax_level_class_names_and_mapping(cate_data_dict)

        split_path = os.path.join(root, SPLIT_FILE[mode])
        base_classes_txt_files = glob(split_path + "/*")
        # Keep it in ascending order
        base_classes_txt_files = sorted(base_classes_txt_files)

        image_path = os.path.join(root, IMAGE_FILE[mode])

        self.samples = []
        for base_class_txt_file in base_classes_txt_files:
            base_class_name = os.path.splitext(os.path.basename(base_class_txt_file))[0]
            base_class_id = int(base_class_name.split(".")[0][-4:])
            base_class_image_path = os.path.join(image_path, f"{base_class_id}")

            
            with open(base_class_txt_file, "r") as f:
                image_names = f.readlines()
            image_names = [image_name.strip() for image_name in image_names]
            image_paths = [
                os.path.join(base_class_image_path, image_name) for image_name in image_names
            ]

            _tmp = np.tile(
                self.class_labels_of_all_tax_levels[:, base_class_id], (len(image_paths), 1)
            )
            self.samples.extend(
                [
                    (image_path, class_labels_of_all_tax_level.tolist())
                    for image_path, class_labels_of_all_tax_level in zip(image_paths, _tmp)
                ]
            )

        self.transform = transform

    def _read_categories(self, cate_path):
        with open(cate_path) as file:
            cate_data = json.load(file)

        # The classification information for each category is stored in a dictionary
        cate_data_dict = {}  
        # Retrieve each piece of data in the order of the cate_data list
        for dt in cate_data:
            dt_id = dt.pop("id")  
            dt_name = dt.pop("name")  
            dt["species"] = dt_name  

            new_dt = []
            # Get the labels for each level in order
            for lv, lv_idx in TAX_LEVELS_MAPPING.items():
                new_dt.append(dt.pop(lv))  
            cate_data_dict[dt_id] = new_dt 

        # 按照 键 升序排列
        cate_data_dict = {key: cate_data_dict[key] for key in sorted(cate_data_dict)}

        return cate_data_dict

    def _get_tax_level_class_names_and_mapping(self, cate_data_dict):
        # Each element of the class_multi_label_list contains the names of 7 levels of categories.
        class_multi_label_list = [
            class_multi_label for class_multi_label in cate_data_dict.values()
        ]
        for tax_level_name, ith in TAX_LEVELS_MAPPING.items():
            ith_tax_level_class_names = list(map(itemgetter(ith), class_multi_label_list))
            uni_names_of_ith_tax_level, class_labels_of_ith_tax_level = np.unique(
                ith_tax_level_class_names, return_inverse=True
            )
            self.uni_names_of_all_tax_levels.append(uni_names_of_ith_tax_level.tolist())
            self.class_labels_of_all_tax_levels.append(class_labels_of_ith_tax_level)
        self.class_labels_of_all_tax_levels = np.stack(self.class_labels_of_all_tax_levels)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, index, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    root = "/media/disk12T/2022-sgh/datasets/iNaturalist2019"

    # ==================== transform ====================
    mean_inat19 = [0.454, 0.474, 0.367]
    std_inat19 = [0.237, 0.230, 0.249]
    normalize_tfs_inat19 = transforms.Normalize(mean=mean_inat19, std=std_inat19)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize_tfs_inat19,
            transforms.RandomResizedCrop(224),
        ]
    )
    # ==================== transform ====================

    train_dataset = iNaturalist_2019(root, mode="test", transform=transform)
    ic(len(train_dataset))
    ic(len(train_dataset.uni_names_of_all_tax_levels))  


    ic(len(train_dataset.uni_names_of_all_tax_levels[TAX_LEVELS_MAPPING["class"]]))  # 9
    ic(train_dataset.uni_names_of_all_tax_levels[TAX_LEVELS_MAPPING["class"]])
    ic(len(train_dataset.uni_names_of_all_tax_levels[TAX_LEVELS_MAPPING["phylum"]]))  # 4
    ic(train_dataset.uni_names_of_all_tax_levels[TAX_LEVELS_MAPPING["phylum"]])
    ic(len(train_dataset.uni_names_of_all_tax_levels[TAX_LEVELS_MAPPING["kingdom"]]))  # 3
    ic(train_dataset.uni_names_of_all_tax_levels[TAX_LEVELS_MAPPING["kingdom"]])



    train_loader = data.DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)
    image, index, target_set = next(iter(train_loader))

    ic(image.shape)
    ic(index)
    ic(target_set)
    import torch

    target = torch.cat(target_set, axis=0).reshape(7, -1)
    ic(target)

    ic(target_set[TAX_LEVELS_MAPPING["species"]])

    print("=" * 20)
