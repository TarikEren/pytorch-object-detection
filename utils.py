from pathlib import Path

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
    # TODO: Grab classes and write them into the .csv file
    coco: COCO = COCO(path_to_coco)
    dataframe = pd.DataFrame()
    temp_bbox_list = []
    bbox_list = []
    img_names = []
    for annotation_index in tqdm(coco.imgToAnns, "Parsing annotations"):
        annotation = coco.imgToAnns[annotation_index]
        img = coco.imgs[annotation[0]["image_id"]]
        img_name = img["file_name"]
        img_names.append(img_name)
        for i in range(len(annotation)):
            temp_bbox_list.append(annotation[i]["bbox"])
        bbox_list.append(temp_bbox_list.copy())
        temp_bbox_list.clear()
        
    print(len(img_names), len(bbox_list))
    dataframe["file_names"] = img_names
    dataframe["box_coordinates"] = (bbox_list)
    return dataframe