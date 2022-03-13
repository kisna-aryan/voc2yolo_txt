import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import sys
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt

label = 'test'
xml_path = f'/home/kisna/FLIR_proj/FLIR_ADAS_v2/datasetv2edit/annotest/{label}/Annotations'
txt_path = f'/home/kisna/FLIR_proj/FLIR_ADAS_v2/datasetv2edit/v2yolodataset/{label}/labels'

# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict


# print(extract_info_from_xml('K:/DL_git/FLIRrgb_OD/FLIRrgb_dataset/train/Annotations/FLIR_00001.xml'))
print(extract_info_from_xml('/home/kisna/FLIR_proj/FLIR_ADAS_v2/datasetv2edit/annotest/train/Annotations/video-2Af3dwvs6YPfwSSf6-frame-000000-imW24bapJsHpTahce.xml'))

# Dictionary that maps class names to IDs
class_name_to_id_mapping = {"person": 0,
                           "bike": 1,
                           "car": 2,
                           "dog": 3,
                           "motor": 4,
                           "bus": 5,
                           "train": 6,
                           "truck": 7,
                           "light": 8,
                           "hydrant": 9,
                           "sign": 10,
                           "dog": 11,
                           "skateboard": 12,
                           "stroller": 13,
                           "scooter": 14,
                           "deer" : 15,
                           "other vehicle": 16}

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
            print("Class not in dict:" , b["class"])
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    # save_file_name = os.path.join(txt_path, info_dict["filename"].replace("jpg", "txt"))
    save_file_name = os.path.join(txt_path, info_dict["filename"].replace("jpg", "txt"))

    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))

    # Get the annotations
annotations = [os.path.join(xml_path, x) for x in os.listdir(xml_path) if x[-3:] == "xml"]
annotations.sort()

# Convert and save the annotations
for ann in tqdm(annotations):
    info_dict = extract_info_from_xml(ann)
    convert_to_yolov5(info_dict)
