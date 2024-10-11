"""KITTI only comes with labels for training images.

1. Split Kitti into train, validation, and test sets.
2. Translate labels to the coco format.
"""


import json
import os
import random


import cv2


DATA_OBJECT_IMAGE_2_PATH = '/mnt/sdb/Datasets/KITTI/data_object_image_2/'
DATA_OBJECT_LABEL_2_PATH = '/mnt/sdb/Datasets/KITTI/training/'
KITTI_CATEGORIES = [
    {
        'id': 0,
        'name': 'car',
        'supercategory': 'vehicle',
    },
    #{
    #    'id': 1,
    #    'name': 'van',
    #    'supercategory': 'vehicle',
    #},
    #{
    #    'id': 2,
    #    'name': 'truck',
    #    'supercategory': 'vehicle',
    #},
    #{
    #    'id': 3,
    #    'name': 'pedestrian',
    #    'supercategory': 'person',
    #},
    #{
    #    'id': 4,
    #    'name': 'person_sitting',
    #    'supercategory': 'person',
    #},
    #{
    #    'id': 5,
    #    'name': 'cyclist',
    #    'supercategory': 'person',
    #},
    #{
    #    'id': 6,
    #    'name': 'tram',
    #    'supercategory': 'vehicle',
    #},
    #{
    #    'id': 7,
    #    'name': 'misc',
    #    'supercategory': 'unknown',
    #},
    #{
    #    'id': 8,
    #    'name': 'dontcare',
    #    'supercategory': 'unknown',
    #}
]

def get_cat_id(class_name):
    for cat in KITTI_CATEGORIES:
        if class_name.lower() == cat['name']:
            return cat['id']
    return -1

if __name__ == '__main__':
    # Get list of all annotated images
    master_list = []
    for f in os.listdir(os.path.join(DATA_OBJECT_LABEL_2_PATH, 'label_2')):
        master_list.append(f.strip('.txt'))
    
    # Shuffle list
    random.seed(0)
    random.shuffle(master_list)

    # Split list
    train_list = master_list[:int(0.8 * len(master_list))]
    val_list = master_list[int(0.8 * len(master_list)):int(0.9 * len(master_list))]
    test_list = master_list[int(0.9 * len(master_list)):]

    # Create coco annotations:
    datasets = [(train_list, 'train'), (val_list, 'val'), (test_list, 'test')]
    for dataset in datasets:
        img_list, name = dataset
        data = {
            'info': {'description': 'KITTI, left camera, 2D objects'},
            'categories': KITTI_CATEGORIES,
            'images': [],
            'annotations': [],
        }
        for img in img_list:
            with open(os.path.join(DATA_OBJECT_LABEL_2_PATH, 'label_2', f'{img}.txt'), 'r') as f:
                count = 0
                for line in f:
                    object_type, truncation, occlusion, alpha, left, top, right, bottom, height, width, length, x, y, z, rotation_y = line.split()
                    if get_cat_id(object_type) < 0 or int(occlusion) > 1:
                        continue
                    data['annotations'].append(
                        {
                            'id': int(img) * 100 + count,
                            'image_id': int(img),
                            'category_id': get_cat_id(object_type),
                            'bbox': [int(float(left)), int(float(top)), int(float(right) - float(left)), int(float(bottom) - float(top))],
                            'iscrowd': False,
                            'area': int(float(right) - float(left)) * int(float(bottom) - float(top)),
                        }
                    )
                    count += 1
                
                frame = cv2.imread(os.path.join(DATA_OBJECT_IMAGE_2_PATH, 'training', 'image_2', f'{img}.png'))
                height, width, _ = frame.shape
                data['images'].append(
                    {
                        'id': int(img),
                        'width': width,
                        'height': height,
                        'file_name': f'{img}.png',
                    }
                )
        
        with open(f'kitti_{name}_annotations.json', 'w') as f:
            f.write(json.dumps(data, indent=4))
