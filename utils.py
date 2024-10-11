import os
import csv
from pathlib import Path
from ast import literal_eval

import torch
import pandas as pd
from tqdm.auto import tqdm
from pycocotools.coco import COCO

def get_classes(path_to_classes: Path) -> list:
    """
    Get all classes from the provided .txt file
    
    Args:
        path_to_classes (Path): Path to the .txt file
        
    Returns:
        (List(str)): A list of classes
    """
    classes_list = []
    with open(path_to_classes, "r") as f:
        raw_classes = f.readlines()
        
    for _class in raw_classes:
        classes_list.append(_class.strip())
        
    return classes_list

def coco_to_dict(path_to_coco: Path) -> pd.DataFrame:
    coco: COCO = COCO(path_to_coco)
    dataframe = pd.DataFrame()
    temp_bbox_list = []
    temp_class_list = []
    class_list = []
    bbox_list = []
    img_names = []
    for annotation_index in tqdm(coco.imgToAnns, "Parsing annotations"):
        annotation = coco.imgToAnns[annotation_index]
        img = coco.imgs[annotation[0]["image_id"]]
        img_name = img["file_name"]
        img_names.append(img_name)
        for i in range(len(annotation)):
            temp_bbox_list.append(literal_eval(annotation[i]["bbox"]))
            temp_class_list.append(literal_eval(annotation[i]["category_id"]))
        bbox_list.append(temp_bbox_list.copy())
        class_list.append(temp_class_list.copy())
        temp_bbox_list.clear()
        temp_class_list.clear()
        
    print(len(img_names), len(bbox_list))
    dataframe["file_names"] = img_names
    dataframe["box_coordinates"] = bbox_list
    dataframe["classes"] = class_list
    return dataframe

def coco_to_csv(coco_json_file: Path, target_path: Path):
    coco = COCO(coco_json_file)
    print("INFO: Checking file")
    imgIds = coco.getImgIds()
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        if not os.path.isfile(os.path.join("../datasets/cup_dataset/train", img['file_name'])) and \
        not os.path.isfile(os.path.join("../datasets/cup_dataset/test", img['file_name'])) and \
        not os.path.isfile(os.path.join("../datasets/cup_dataset/cup", img['file_name'])):
            print('The image {} does not exist.'.format(img['file_name']))
            exit()

    csv_file = open(target_path, 'w')
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)

    print('INFO: Writing annotations file')

    annIds = coco.getAnnIds()
    for annId in annIds:
        ann = coco.loadAnns(annId)[0]
        img = coco.loadImgs(ann['image_id'])[0]
        cat = coco.loadCats(ann['category_id'])[0]
            
        x1,y1,w,h = ann['bbox']
        x2 = x1 + w
        y2 = y1 + h
        
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        
        csv_writer.writerow([os.path.join(coco_json_file, img['file_name']), str(x1), str(y1), str(x2), str(y2), cat['name']])

    csv_file.close()

    print('INFO: Annotations file was written')

    csv_class_file = open("classes.txt", 'w')
    csv_class_writer = csv.writer(csv_class_file, quoting=csv.QUOTE_MINIMAL)

    print('INFO: Writing classes file')

    catIds = coco.getCatIds()
    nowId = 1
    for catId in catIds:
        cat = coco.loadCats(catId)[0]    
            
        csv_class_writer.writerow([cat['name'], str(nowId)])
        nowId = nowId + 1
        
    csv_class_file.close()

    print("INFO: Classes file was written")

def get_coco_annotations(output_file: Path, coco: COCO) -> list:
    """
    Reads a COCO file and creates a text file to read from in order to create an annotation list
    
    Args:
        output_file (Path): Where the text file should be created
        coco (COCO): COCO instance to read from
    
    Returns:
        List[str, List[List[float], int]]: The list of image names and their corresponding annotations.
    """
    
    # Check if output file already exists to get rid of unnecessary file creations.
    if output_file.exists():
        print("INFO: Text file already exists, skipping file generation")
    
    # If it doesn't exist
    else:
        # Get all image ids
        img_ids: list = coco.getImgIds()
        
        # Open the output text file
        with open(output_file, "w") as text_file:
            # For every id
            for id in img_ids:
                # Get the annotations
                anns = coco.anns[id]
                
                # Create an annotation list
                ann_list = [anns["bbox"], anns["category_id"]]
                
                # Get the image name using the image_id property of the annotation dictionary
                img = coco.loadImgs(anns["image_id"])[0]["file_name"]
                
                # Write the image and its corresponding annotations into the output file
                text_file.write(f"[\"{img}\", {ann_list}]\n")
                
    # Open the output file of the function           
    with open(output_file, "r") as f:
        # Read the lines from it and create a list
        ann_list = f.readlines()
        
        # Create a list to hold the to-be-transformed values
        annotation_list = []
        
        # For every annotation
        for ann in ann_list:
            # Strip and append the annotation to the annotation_list we've just created
            annotation_list.append(literal_eval(ann.strip()))
    
    # Return the annotation list
    return annotation_list

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized inputs in the DataLoader.

    Args:
        batch (list): A list of tuples (image_tensor, labels).

    Returns:
        Tuple(torch.Tensor, List): A tuple containing a batch of images and a list of labels.
    """
    # Separate the images and the labels
    images, labels = zip(*batch)

    # Stack images into a single tensor
    images = torch.stack(images)

    # Initialize lists for boxes and classes
    boxes = []
    classes = []

    # Iterate through each label and append to respective lists
    for label in labels:
        # TODO: Fix as this approach may fail with frames that contain more or less than 2 frames
        boxes.append(label[0][0])   # each is a tensor of shape (N, 4)
        boxes.append(label[1][0])
        classes.append(label[0][1]) # each is a tensor of shape (N,)
        classes.append(label[1][1])

    return images, {'boxes': boxes, 'classes': classes}
