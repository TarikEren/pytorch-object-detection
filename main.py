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

model_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=model_weights)

def main():
    # TODO: Make the user provide the path using args.    
    print(device, model)
    
# Main function caller
if __name__ == "__main__":
    main()