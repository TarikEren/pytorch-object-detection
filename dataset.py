from pathlib import Path

import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from PIL import Image

class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_to_dataset: Path,
                 transforms: torchvision.transforms,
                 annotation_list: list) -> None:
        """
        Initialises the dataset
        
        Args:
            path_to_dataset (Path): Path to the dataset
            transforms (torchvision.transforms): Transforms to be applied to the image
            annotation_list (list): List of images and their corresponding annotations
        """
       
        # Initialise the variables
        self.path_to_dataset: Path = path_to_dataset,
        self.transforms: torchvision.transforms = transforms
        self.annotation_list: list = annotation_list
        self.images: list = list(path_to_dataset.glob("*.png"))
    
    def __len__(self):
        """
        Returns the amount of images
        
        Returns:
            (int): The image count
        """
        return len(self.images)
        
    def __getitem__(self, index):
        # If no images found, return None
        if self.__len__() == 0:
            print("ERROR: No images found. Aborting")
            return None
        annotations = []
        # Turn the extension uppercase as COCO stores extensions in uppercase
        image_name = self.images[index].name.rstrip(".png") + ".PNG"
        for entry in self.annotation_list:
            if entry[0] == image_name:
                image_path = self.path_to_dataset[0] / entry[0]
                annotations.append(entry[1])
        
        image = Image.open(image_path)
        image = self.transforms(image)
        return image, annotations