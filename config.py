from pathlib import Path


import torch
import torchvision
import torchvision.transforms.v2
from pycocotools.coco import COCO

# Temporary paths
image_path: Path = Path("../datasets/frames_and_annotations")
train_path: Path = image_path / "train"
csv_path: Path = Path("out.csv")
path_to_coco: Path = Path("coco.json")
output_text_path: Path = Path("out.txt")
output_csv_path: Path = Path("out.csv")
coco = COCO(path_to_coco)

# Constants / Hyperparameters
EPOCHS = 10
BATCH_SIZE = 1
NUM_WORKERS = 1
LEARNING_RATE = 0.001


# Set the device based on the available devices in the system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and its weights
model_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=model_weights)

# Transforms for the training data.
train_transform = torchvision.transforms.v2.Compose(transforms=[
    torchvision.transforms.v2.Compose(transforms=[torchvision.transforms.v2.ToImage(), torchvision.transforms.v2.ToDtype(torch.float32, scale=True)]),
    torchvision.transforms.v2.Resize(size=(224,224)),
    torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.v2.RandomVerticalFlip(p=0.5),
])

# Transforms for the testing data.
test_transform = torchvision.transforms.Compose(transforms=[
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
])

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)