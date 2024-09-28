from pathlib import Path
from ast import literal_eval

import pandas as pd
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class to be used with a torch.utils.data.dataloader.
    """
    def __init__(self, 
                 image_path: Path,
                 dataframe: pd.DataFrame,
                 transforms: torchvision.transforms):
        
        self.dataframe = dataframe
        self.transforms = transforms
        self.image_path = image_path

    def __len__(self):
        """
        Returns the length of the dataset.
        
        Returns:
            (int): Length of the dataset
        """
        return len(self.dataframe)
        
        
    def __getitem__(self, idx):
        """
        Gets an image using the index and returns said image with its annotations.
        
        Args:
            idx (int): Index
        
        Returns:
            Tuple(torch.Tensor, torch.Tensor): A tuple of tensorized image with its annotations
        """
        image_name: str = self.dataframe["file_names"][idx]
        
        try:
            image: Image = Image.open(self.image_path / image_name)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image {image_name} not found at path {self.image_path}")
        
        try:        
            boxes = literal_eval(self.dataframe.iloc[idx]["box_coordinates"])
            classes = literal_eval(self.dataframe.iloc[idx]["classes"])
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Error parsing annotations for index {idx}: {e}")
        
        labels = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "classes": torch.tensor(classes, dtype=torch.int64)
        }
        
        image_tensor = self.transforms(image)
        
        return image_tensor, labels