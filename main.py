# TODO: Make the user provide paths using args.

from pathlib import Path

import torch
import torchvision
import pandas as pd

from utils import coco_to_dict
from dataset import Dataset

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

# Set the device based on the available devices in the system.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and its weights
model_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=model_weights)

# Temporary paths
image_path: Path = Path("../datasets/cup_dataset/train")
csv_path: Path = Path("out.csv")
dataframe = pd.read_csv(csv_path)

def main():
    
    
    dataset = Dataset(image_path=image_path,
                      dataframe=dataframe,
                      transforms=train_transform)
           
    
# Main function caller
if __name__ == "__main__":
    main()