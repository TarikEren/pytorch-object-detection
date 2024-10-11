# TODO: Write a testing loop
# TODO: Write a training loop
# TODO: Write a drawing function
# TODO: Make the user provide paths using args.

from ast import literal_eval

import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from tqdm.auto import tqdm

from dataset import Custom_Dataset
from utils import collate_fn, get_coco_annotations
from config import (output_text_path,
                    train_path,
                    coco,
                    EPOCHS,
                    BATCH_SIZE,
                    NUM_WORKERS,
                    model,
                    device,
                    train_transform)



def main():
    annotation_list = get_coco_annotations(output_file=output_text_path, coco=coco)
    train_dataset = Custom_Dataset(path_to_dataset=train_path, transforms=train_transform, annotation_list=annotation_list)
    
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=BATCH_SIZE,
                                                num_workers=NUM_WORKERS,
                                                collate_fn=collate_fn)

    model.to(device=device)
    # Set the model to training mode.
    
    model.train()

    
    for epoch in tqdm(range(EPOCHS), "Training model"):
        
        # Initialise the loss as 0.0
        loss = 0.0
        
        for images, targets in train_dataloader:
            # Send the images to the device
            images = [image.to(device) for image in images]

            # Send the targets to device
            boxes = torch.tensor(targets["boxes"]).to(device)
            classes = torch.tensor(targets["classes"]).to(device)
            targets = {
                "boxes": boxes,
                "classes": classes
            }

# Main function caller
if __name__ == "__main__":
    main()