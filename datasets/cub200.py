import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class CUB200(Dataset):
    @classmethod
    def fine_to_coarse(cls, fine_classes, fine_class_to_idx):
        """
        Args:
            fine_classes (List[str]): 200 fine-grained species names.
            fine_class_to_idx (Dict[str, int]): mapping from fine name to its index.

        Returns:
            Dict[int, int]: mapping from fine index to coarse index.
        """
        # 1) Define 24 coarse categories and their sequence (coarse idx = list index)
        coarse_classes = [
            "Waterbird",
            "Icterid",
            "Finch",
            "Grosbeak/Cardinal",
            "Bunting",
            "Sparrow",
            "Warbler/Chat",
            "Flycatcher",
            "Vireo",
            "Wren",
            "Thrasher/Mimid",
            "Nightjar",
            "Hummingbird",
            "Woodpecker",
            "Corvid",
            "Cuculidae",
            "Swallow",
            "Tanager",
            "Pipit",
            "Bark_Forager",
            "Shrike",
            "Kingfisher",
            "Starling",
            "Waxwing",
        ]

        coarse_class_to_idx = {name: idx for idx, name in enumerate(coarse_classes)}

        # 2) Automatically construct coarse_mapping based on name keywords
        coarse_mapping = {
            # for example: 1. Waterbirds (Seabirds and waterfowl)
            "Waterbird": [
                "Black_Footed_Albatross",
                "Laysan_Albatross",
                "Sooty_Albatross",
                "Crested_Auklet",
                "Least_Auklet",
                "Parakeet_Auklet",
                "Rhinoceros_Auklet",
                "Horned_Puffin",
                "Pigeon_Guillemot",
                "Brandt_Cormorant",
                "Red_Faced_Cormorant",
                "Pelagic_Cormorant",
                "Eared_Grebe",
                "Horned_Grebe",
                "Pied_Billed_Grebe",
                "Western_Grebe",
                "Brown_Pelican",
                "White_Pelican",
                "Frigatebird",
                "Northern_Fulmar",
                "Long_Tailed_Jaeger",
                "Pomarine_Jaeger",
                "California_Gull",
                "Glaucous_Winged_Gull",
                "Heermann_Gull",
                "Herring_Gull",
                "Ivory_Gull",
                "Ring_Billed_Gull",
                "Slaty_Backed_Gull",
                "Western_Gull",
                "Artic_Tern",
                "Black_Tern",
                "Caspian_Tern",
                "Common_Tern",
                "Elegant_Tern",
                "Forsters_Tern",
                "Least_Tern",
                "Pacific_Loon",
                "Gadwall",
                "Mallard",
                "Hooded_Merganser",
                "Red_Breasted_Merganser",
                "Red_Legged_Kittiwake",  
            ],
            # 2. Icterids (Finches - blackbirds, bullbirds, grackles, orioles, ravens, etc)
            "Icterid": [
                "Brewer_Blackbird",
                "Red_Winged_Blackbird",
                "Rusty_Blackbird",
                "Yellow_Headed_Blackbird",
                "Bobolink",
                "Bronzed_Cowbird",
                "Shiny_Cowbird",
                "Boat_Tailed_Grackle",
                "Baltimore_Oriole",
                "Hooded_Oriole",
                "Orchard_Oriole",
                "Scott_Oriole",
                "Western_Meadowlark",
            ],
            # 3. Finches (Finches - the goldfinch and rosefinch of the passerine order)
            "Finch": [
                "American_Goldfinch",
                "European_Goldfinch",
                "Gray_Crowned_Rosy_Finch",
                "Purple_Finch",
            ],
            # 4. Grosbeaks & Cardinals (Finches - Toucans and cardinals)
            "Grosbeak/Cardinal": [
                "Blue_Grosbeak",
                "Evening_Grosbeak",
                "Pine_Grosbeak",
                "Rose_Breasted_Grosbeak",
                "Cardinal",
            ],
            # 5. Buntings (Fringillidae - Genus Fringillidae)
            "Bunting": ["Indigo_Bunting", "Lazuli_Bunting", "Painted_Bunting"],
            # 6. Sparrows & Ground-foragers (Finidae - ground-dwelling sparrows and ground-singing birds)
            "Sparrow": [
                "Baird_Sparrow",
                "Black_Throated_Sparrow",
                "Brewer_Sparrow",
                "Chipping_Sparrow",
                "Clay_Colored_Sparrow",
                "House_Sparrow",
                "Field_Sparrow",
                "Fox_Sparrow",
                "Grasshopper_Sparrow",
                "Harris_Sparrow",
                "Henslow_Sparrow",
                "Le_Conte_Sparrow",
                "Lincoln_Sparrow",
                "Nelson_Sharp_Tailed_Sparrow",
                "Savannah_Sparrow",
                "Seaside_Sparrow",
                "Song_Sparrow",
                "Tree_Sparrow",
                "Vesper_Sparrow",
                "White_Crowned_Sparrow",
                "White_Throated_Sparrow",
                "Dark_Eyed_Junco",
                "Horned_Lark",
                "Eastern_Towhee",  
                "Green_Tailed_Towhee",  
            ],
            # 7. Warblers & Chats 
            "Warbler/Chat": [
                "Bay_Breasted_Warbler",
                "Black_And_White_Warbler",
                "Black_Throated_Blue_Warbler",
                "Blue_Winged_Warbler",
                "Canada_Warbler",
                "Cape_May_Warbler",
                "Cerulean_Warbler",
                "Chestnut_Sided_Warbler",
                "Golden_Winged_Warbler",
                "Hooded_Warbler",
                "Kentucky_Warbler",
                "Magnolia_Warbler",
                "Mourning_Warbler",
                "Myrtle_Warbler",
                "Nashville_Warbler",
                "Orange_Crowned_Warbler",
                "Palm_Warbler",
                "Pine_Warbler",
                "Prairie_Warbler",
                "Prothonotary_Warbler",
                "Swainson_Warbler",
                "Tennessee_Warbler",
                "Wilson_Warbler",
                "Worm_Eating_Warbler",
                "Yellow_Warbler",
                "Northern_Waterthrush",
                "Louisiana_Waterthrush",
                "Common_Yellowthroat",
                "American_Redstart",
                "Ovenbird",
                "Yellow_Breasted_Chat",
            ],
            # 8. Flycatchers & Pewees 
            "Flycatcher": [
                "Acadian_Flycatcher",
                "Great_Crested_Flycatcher",
                "Least_Flycatcher",
                "Olive_Sided_Flycatcher",
                "Scissor_Tailed_Flycatcher",
                "Vermilion_Flycatcher",
                "Yellow_Bellied_Flycatcher",
                "Western_Wood_Pewee",
                "Sayornis",
                "Tropical_Kingbird", 
                "Gray_Kingbird",  
            ],
            # 9. Vireos 
            "Vireo": [
                "Black_Capped_Vireo",
                "Blue_Headed_Vireo",
                "Philadelphia_Vireo",
                "Red_Eyed_Vireo",
                "Warbling_Vireo",
                "White_Eyed_Vireo",
                "Yellow_Throated_Vireo",
            ],
            # 10. Wrens 
            "Wren": [
                "Bewick_Wren",
                "Cactus_Wren",
                "Carolina_Wren",
                "House_Wren",
                "Marsh_Wren",
                "Rock_Wren",
                "Winter_Wren",
            ],
            # 11. Thrashers & Mimids 
            "Thrasher/Mimid": [
                "Spotted_Catbird",
                "Gray_Catbird",
                "Brown_Thrasher",
                "Sage_Thrasher",
                "Mockingbird",
            ],
            # 12. Nightjars 
            "Nightjar": ["Chuck_Will_Widow", "Nighthawk", "Whip_Poor_Will"],
            # 13. Hummingbirds 
            "Hummingbird": [
                "Anna_Hummingbird",
                "Ruby_Throated_Hummingbird",
                "Rufous_Hummingbird",
                "Green_Violetear",
            ],
            # 14. Woodpeckers & Flicker 
            "Woodpecker": [
                "American_Three_Toed_Woodpecker",
                "Pileated_Woodpecker",
                "Red_Bellied_Woodpecker",
                "Red_Cockaded_Woodpecker",
                "Red_Headed_Woodpecker",
                "Downy_Woodpecker",
                "Northern_Flicker",
            ],
            # 15. Corvids 
            "Corvid": [
                "American_Crow",
                "Fish_Crow",
                "Common_Raven",
                "White_Necked_Raven",
                "Blue_Jay",
                "Florida_Jay",
                "Green_Jay",
                "Clark_Nutcracker",
            ],
            # 16. Cuculids 
            "Cuculidae": [
                "Black_Billed_Cuckoo",
                "Mangrove_Cuckoo",
                "Yellow_Billed_Cuckoo",
                "Groove_Billed_Ani",
                "Geococcyx",
            ],
            # 17. Swallows 
            "Swallow": [
                "Bank_Swallow",
                "Barn_Swallow",
                "Cliff_Swallow",
                "Tree_Swallow",
            ],
            # 18. Tanagers 
            "Tanager": ["Scarlet_Tanager", "Summer_Tanager"],
            # 19. Pipits 
            "Pipit": ["American_Pipit"],
            # 20. Bark-foragers 
            "Bark_Forager": ["White_Breasted_Nuthatch", "Brown_Creeper"],
            # 21. Shrikes 
            "Shrike": ["Loggerhead_Shrike", "Great_Grey_Shrike"],
            # 22. Kingfishers 
            "Kingfisher": [
                "Belted_Kingfisher",
                "Green_Kingfisher",
                "Pied_Kingfisher",
                "Ringed_Kingfisher",
                "White_Breasted_Kingfisher",
            ],
            # 23. Starlings 
            "Starling": ["Cape_Glossy_Starling"],
            # 24. Waxwings 
            "Waxwing": ["Bohemian_Waxwing", "Cedar_Waxwing"],
        }

        # 3) fine_name -> coarse_name
        fine_to_coarse = {
            fine_name: coarse_name
            for coarse_name, fine_list in coarse_mapping.items()
            for fine_name in fine_list
        }

        # 4) fine_idx -> coarse_idx
        fine_idx_to_coarse_idx = {}
        for fine_name, fine_idx in fine_class_to_idx.items():
            coarse_name = fine_to_coarse.get(fine_name)
            coarse_idx = coarse_class_to_idx[coarse_name]
            fine_idx_to_coarse_idx[fine_idx] = coarse_idx

        return fine_idx_to_coarse_idx

    def __init__(self, path, train=True, transform=None, target_transform=None):
        self.root = path
        self.is_train = train
        self.transform = transform
        self.target_transform = target_transform

        # self.cls_num_each_level = [200]

        # Load image paths
        self.images_path = {}
        with open(os.path.join(self.root, "images.txt")) as f:
            for line in f:
                image_id, path = line.strip().split()
                self.images_path[image_id] = path

        def capitalize_each_part(s: str) -> str:
            return "_".join(part.capitalize() for part in s.split("_"))

        # Load classes
        self.classes = []
        self.class_to_idx = {}
        with open(os.path.join(self.root, "classes.txt")) as f:
            for line in f:
                idx, class_name = line.strip().split()
                class_name = class_name.split(".")[1]
                class_name = capitalize_each_part(class_name)
                self.classes.append(class_name)
                self.class_to_idx[class_name] = int(idx) - 1

        with open("classes.txt", "w") as f:
            for c in self.classes:
                f.write(f"{c}\n")

        self.fine_idx_to_coarse_idx = CUB200.fine_to_coarse(
            self.classes, self.class_to_idx
        )

        # Load class labels
        self.class_ids = {}
        with open(os.path.join(self.root, "image_class_labels.txt")) as f:
            for line in f:
                image_id, class_id = line.strip().split()
                self.class_ids[image_id] = class_id

        # Split train/test
        self.data_id = []
        split_file = os.path.join(self.root, "train_test_split.txt")
        with open(split_file) as f:
            for line in f:
                image_id, is_train = line.strip().split()
                if (self.is_train and int(is_train)) or (
                    not self.is_train and not int(is_train)
                ):
                    self.data_id.append(image_id)

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, idx):
        """
        Args:
            index: index of training dataset
        Returns:
            image and its corresponding label
        """
        image_id = self.data_id[idx]
        class_id = int(self._get_class_by_id(image_id)) - 1  # convert to 0-based index
        path = self._get_path_by_id(image_id)

        # Load image with Pillow
        image = Image.open(os.path.join(self.root, "images", path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            class_id = self.target_transform(class_id)

        return image, idx, self.fine_idx_to_coarse_idx[class_id], class_id

    def _get_path_by_id(self, image_id):
        return self.images_path[image_id]

    def _get_class_by_id(self, image_id):
        return self.class_ids[image_id]


if __name__ == "__main__":
    root = "/media/ssd2T/Datasets/cub200/CUB_200_2011/CUB_200_2011"
    dataset = CUB200(path=root, train=True)

    print(len(dataset.classes))
    print(dataset.classes)
    print(dataset[0])
