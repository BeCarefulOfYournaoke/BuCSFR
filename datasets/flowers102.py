from torchvision.datasets import Flowers102


class MyFlowers102(Flowers102):
    fine_names = [
        "pink primrose",
        "hard-leaved pocket orchid",
        "canterbury bells",
        "sweet pea",
        "english marigold",
        "tiger lily",
        "moon orchid",
        "bird of paradise",
        "monkshood",
        "globe thistle",
        "snapdragon",
        "colt's foot",
        "king protea",
        "spear thistle",
        "yellow iris",
        "globe-flower",
        "purple coneflower",
        "peruvian lily",
        "balloon flower",
        "giant white arum lily",
        "fire lily",
        "pincushion flower",
        "fritillary",
        "red ginger",
        "grape hyacinth",
        "corn poppy",
        "prince of wales feathers",
        "stemless gentian",
        "artichoke",
        "sweet william",
        "carnation",
        "garden phlox",
        "love in the mist",
        "mexican aster",
        "alpine sea holly",
        "ruby-lipped cattleya",
        "cape flower",
        "great masterwort",
        "siam tulip",
        "lenten rose",
        "barbeton daisy",
        "daffodil",
        "sword lily",
        "poinsettia",
        "bolero deep blue",
        "wallflower",
        "marigold",
        "buttercup",
        "oxeye daisy",
        "common dandelion",
        "petunia",
        "wild pansy",
        "primula",
        "sunflower",
        "pelargonium",
        "bishop of llandaff",
        "gaura",
        "geranium",
        "orange dahlia",
        "pink-yellow dahlia",
        "cautleya spicata",
        "japanese anemone",
        "black-eyed susan",
        "silverbush",
        "californian poppy",
        "osteospermum",
        "spring crocus",
        "bearded iris",
        "windflower",
        "tree poppy",
        "gazania",
        "azalea",
        "water lily",
        "rose",
        "thorn apple",
        "morning glory",
        "passion flower",
        "lotus",
        "toad lily",
        "anthurium",
        "frangipani",
        "clematis",
        "hibiscus",
        "columbine",
        "desert-rose",
        "tree mallow",
        "magnolia",
        "cyclamen",
        "watercress",
        "canna lily",
        "hippeastrum",
        "bee balm",
        "ball moss",
        "foxglove",
        "bougainvillea",
        "camellia",
        "mallow",
        "mexican petunia",
        "bromelia",
        "blanket flower",
        "trumpet creeper",
        "blackberry lily",
    ]

    @classmethod
    def fine_to_coarse(cls, fine_names, fine_to_idx):
        """
        Args:
            fine_names (List[str]): 102 Oxford-102 fine-grained flower names.
            fine_to_idx (Dict[str, int]): mapping from fine name to its index (0-101).

        Returns:
            Dict[int, int]: mapping from fine index to coarse index (0-3).
        """
        # 1) Define the coarse-grained categories and their order
        coarse_names = [
            "Orchids & Alstroemeria",
            "Lilies & Arums",
            "Irises, Crocus & Allies",
            "Daisies & Sunflowers",
            "Marigolds & Thistles",
            "Poppies",
            "Bellflowers & Tubular",
            "Roses & Mallows",
            "Primroses & Phloxes",
            "Aquatics & Spathes",
            "Tropicals & Vines",
            "Others",
        ]
        coarse_to_idx = {name: i for i, name in enumerate(coarse_names)}

        # 2) Construct a coarse-grained mapping
        coarse_mapping = {
            # 1. Orchids & Alstroemeria: Flowers with distinct lip petals and complex structures
            "Orchids & Alstroemeria": [
                "hard-leaved pocket orchid",
                "moon orchid",
                "ruby-lipped cattleya",
                "peruvian lily",
                "cautleya spicata",
                "red ginger",  # The flower structure of the Zingiberaceae family is complex and is related to the Orchidaceae family
                "cape flower",  # The Proteaceae family, morphologically similar to the Orchidaceae family
            ],
            # 2. Liliaceae and araceae (Lilies & Arums) : 6-leafed bracts or large bracts
            "Lilies & Arums": [
                "tiger lily",
                "giant white arum lily",
                "fire lily",
                "sword lily",
                "canna lily",
                "hippeastrum", 
            ],
            # 3. Iridaceae and bulbous flowers (Irises, Crocus & Allies) : 6-petal perianth, sword leaves
            "Irises, Crocus & Allies": [
                "yellow iris",
                "bearded iris",
                "grape hyacinth",
                "spring crocus",
                "blackberry lily",  # Iris Special Feature
            ],
            # 4. Compositae - Daisies & Sunflowers: Radiating flowers + disc-shaped inflorescences
            "Daisies & Sunflowers": [
                "oxeye daisy",
                "barbeton daisy",
                "osteospermum",
                "gazania",
                "blanket flower",
                "black-eyed susan",
                "sunflower",
                "king protea",  # The Shanlong Ophthalmology family has a morphology similar to that of the Asteraceae family
            ],
            # 5. Compositae - Marigolds & Thistles
            "Marigolds & Thistles": [
                "english marigold",
                "marigold",
                "pincushion flower",
                "artichoke",
                "globe thistle",
                "spear thistle",
                "alpine sea holly",
                "globe-flower", 
            ],
            # 6. Poppies
            "Poppies": [
                "corn poppy",
                "californian poppy",
                "tree poppy",
            ],
            # 7. Bellflowers & Tubular
            "Bellflowers & Tubular": [
                "canterbury bells",
                "snapdragon",
                "foxglove",
                "balloon flower",
                "columbine",
                "toad lily",
                "japanese anemone",
                "monkshood",
                "bee balm",  
            ],
            # 8. Roses & Mallows
            "Roses & Mallows": [
                "rose",
                "desert-rose",
                "hibiscus",
                "camellia",
                "mallow",
                "magnolia",  
            ],
            # 9. Primroses & Phloxes
            "Primroses & Phloxes": [
                "pink primrose",
                "primula",
                "colt's foot",
                "lenten rose",
                "garden phlox",
                "sweet william",
                "wallflower",
                "gaura",
                "carnation", 
            ],
            # 10. Aquatics & Spathes
            "Aquatics & Spathes": [
                "water lily",
                "lotus",
                "watercress",
                "anthurium",
                "cyclamen",
            ],
            # 11. Tropicals & Vines
            "Tropicals & Vines": [
                "bird of paradise",
                "morning glory",
                "passion flower",
                "clematis",
                "trumpet creeper",  
                "bromelia",
                "mexican petunia",
                "tree mallow",
                "bougainvillea",  
                "petunia",
                "pelargonium",
                "geranium",
                "bishop of llandaff",
                "prince of wales feathers",
                "frangipani",  
            ],
            # 12. Others
            "Others": [
                "monkshood",  
                "fritillary",
                "purple coneflower",
                "stemless gentian",
                "love in the mist",
                "mexican aster",
                "great masterwort",
                "siam tulip",
                "daffodil",
                "poinsettia",
                "bolero deep blue",
                "buttercup",
                "common dandelion",
                "wild pansy",
                "orange dahlia",
                "pink-yellow dahlia",
                "silverbush",
                "azalea",
                "thorn apple",
                "ball moss",
                "windflower",
                "sweet pea",
            ],
        }

        # 3) Build a reverse mapping: fine-grained names → coarse-grained names
        fine_to_coarse = {
            fine: coarse for coarse, fines in coarse_mapping.items() for fine in fines
        }

        # 4) Final mapping: Fine-grained index → coarse-grained index
        fine_idx_to_coarse_idx = {}
        for fine, idx in fine_to_idx.items():
            coarse = fine_to_coarse.get(fine)

            assert coarse is not None

            coarse_idx = coarse_to_idx[coarse]
            fine_idx_to_coarse_idx[idx] = coarse_idx

        return fine_idx_to_coarse_idx

    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        super().__init__(root, split, transform, target_transform, download)

        fine_names = MyFlowers102.fine_names
        fine_to_idx = {}
        for idx, fine_name in enumerate(fine_names):
            fine_to_idx[fine_name] = idx
        self.fine_idx_to_coarse_idx = MyFlowers102.fine_to_coarse(
            fine_names, fine_to_idx
        )

    def __getitem__(self, idx):
        img, lab = super().__getitem__(idx)
        return img, idx, self.fine_idx_to_coarse_idx[lab], lab


if __name__ == "__main__":
    root = "/media/ssd2T/Datasets/flowers102"
    dataset = MyFlowers102(root=root, split="train")

    print(len(dataset))
