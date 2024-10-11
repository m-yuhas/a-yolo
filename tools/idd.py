"""Change the IDD dataset to Coco labels."""

import json
import logging
import os
from xml.etree import ElementTree


import coloredlogs


LOGGER = logging.getLogger('Data Conversion')
coloredlogs.install(level='DEBUG')


DATASET_ROOT = '/mnt/sdb/Datasets/IDD/IDD_Detection/'
DATA_PATH = 'JPEGImages'
LABEL_PATH = 'Annotations'
IDD_CATEGORIES = [
    {
        'id': 0,
        'name': 'bicycle',
        'supercategory': 'vehicle',
    },
    {
        'id':  1,
        'name': 'bus',
        'supercategory': 'vehicle',
    },
    {
        'id': 2,
        'name': 'traffic sign',
        'supercategory': 'infrastructure',
    },
    {
        'id': 3,
        'name': 'motorcycle',
        'supercategory': 'vehicle',
    },
    {
        'id': 4,
        'name': 'car',
        'supercategory': 'vehicle',
    },
    {
        'id': 5,
        'name': 'traffic light',
        'supercategory': 'infrastructure',
    },
    {
        'id': 6,
        'name': 'person',
        'supercategory': 'human',
    },
    {
        'id': 7,
        'name': 'vehicle fallback',
        'supercategory': 'vehicle',
    },
    {
        'id': 8,
        'name': 'truck',
        'supercategory': 'vehicle',
    },
    {
        'id': 9,
        'name': 'autorickshaw',
        'supercategory': 'vehicle',
    },
    {
        'id': 10,
        'name': 'animal',
        'supercategory': 'animal',
    },
    {
        'id': 11,
        'name': 'rider',
        'supercategory': 'human'
    },
    {
        'id': 12,
        'name': 'train',
        'supercategory': 'vehicle',
    },
    {
        'id': 13,
        'name': 'caravan',
        'supercategory': 'vehicle',
    },
    {
        'id': 14,
        'name': 'trailer',
        'supercategory': 'vehicle',
    },
]

def get_cat_id(class_name):
    for cat in IDD_CATEGORIES:
        if class_name.lower() == cat['name']:
            return cat['id']
    raise Exception('Category name not in dataset')


if __name__ == '__main__':
    for dataset in ['test', 'train', 'val']:
        # Get list of all annotated images
        LOGGER.info(f'Working on {dataset} dataset...')
        img_list = []
        with open(os.path.join(DATASET_ROOT, f'{dataset}.txt'), 'r') as f:
            for line in f:
                img_list.append(line.strip())

        # General dataset info
        data = {
            'info': {'description': 'IDD, Object detection'},
            'categories': IDD_CATEGORIES,
            'images': [],
            'annotations': [],
        }

        # Convert XML to Coco
        img_cnt = 0
        for img in img_list:
            try:
                tree = ElementTree.parse(os.path.join(DATASET_ROOT, LABEL_PATH, f'{img}.xml'))
            except FileNotFoundError:
                LOGGER.warning(f'File: {img} not found!')
                continue

            # Add new image
            data['images'].append(
                {
                    'id': img_cnt,
                    'width': int(tree.find('size').find('width').text),
                    'height': int(tree.find('size').find('height').text),
                    'file_name': f'{img}.jpg',
                }
            )

            # Add object annotations
            obj_cnt = 0
            for obj in tree.findall('object'):
                category = obj.find('name').text
                try:
                    xmin = int(obj.find('bndbox').find('xmin').text)
                    xmax = int(obj.find('bndbox').find('xmax').text)
                    ymin = int(obj.find('bndbox').find('ymin').text)
                    ymax = int(obj.find('bndbox').find('ymax').text)
                    data['annotations'].append(
                        {
                            'id': img_cnt * 100 + obj_cnt,
                            'image_id': img_cnt,
                            'category_id': get_cat_id(category),
                            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                            'iscrowd': False,
                            'area': (xmax - xmin) * (ymax - ymin),
                        }
                    )
                    obj_cnt += 1
                except:
                    LOGGER.warning(f'Unknown Category: {category} appeared')
            
            img_cnt += 1
        
        LOGGER.info(f'Writing {dataset} dataset to file...')
        LOGGER.info(f'This dataset contains: {img_cnt} images.')
        with open(f'idd_{dataset}_annotations.json', 'w') as f:
            f.write(json.dumps(data))
