from pathlib import Path

import torch
import torchvision

from utils import coco_to_dict

# Set the device based on the available devices in the system.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms for the training data.
train_transform = torchvision.transforms.Compose(transforms=[
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.TrivialAugmentWide()
])

# Transforms for the testing data.
test_transform = torchvision.transforms.Compose(transforms=[
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
])


def main():
    # TODO: Make the user provide the path using args.    
    path_to_coco: Path = Path("../datasets/cup_dataset/coco.json") # Temporary
    ann_dict = coco_to_dict(path_to_coco=path_to_coco)
    ann_dict.to_csv("out.csv")
    
# Main function caller
if __name__ == "__main__":
    main()