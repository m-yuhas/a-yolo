"""Change the GERALD dataset to Coco labels."""


import copy
import json
import logging
import os
import random
from xml.etree import ElementTree


import coloredlogs


LOGGER = logging.getLogger('Data Conversion')
coloredlogs.install(level='DEBUG')


DATASET_ROOT = '/mnt/sdb/Datasets/GERALD/dataset/'
DATA_PATH = 'JPEGImages'
LABEL_PATH = 'Annotations'
GERALD_CATEGORIES = []


def get_cat_id(class_name):
    for cat in GERALD_CATEGORIES:
        if class_name == cat['name']:
            return cat['id']
    return -1


if __name__ == '__main__':
    
    # General dataset info
    data = {
        'info': {'description': 'IDD, Object detection'},
        'images': [],
        'annotations': [],
    }

    # Convert XML to Coco
    img_cnt = 0
    for img in os.listdir(os.path.join(DATASET_ROOT, LABEL_PATH)):
        tree = ElementTree.parse(os.path.join(DATASET_ROOT, LABEL_PATH, img))

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
            if get_cat_id(category) < 0:
                GERALD_CATEGORIES.append(
                    {
                        'id': len(GERALD_CATEGORIES),
                        'name': category,
                        'supercategory': 'signal',
                    }
                )
                LOGGER.info(f'Added new category: {category}.')
            xmin = int(float(obj.find('bndbox').find('xmin').text))
            xmax = int(float(obj.find('bndbox').find('xmax').text))
            ymin = int(float(obj.find('bndbox').find('ymin').text))
            ymax = int(float(obj.find('bndbox').find('ymax').text))
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
        img_cnt += 1


    # Split dataset
    LOGGER.info('Splitting dataset...')
    data['categories'] = GERALD_CATEGORIES
    random.shuffle(data['annotations'])

    # Save Dataset
    for ds, split in zip(['train', 'val', 'test'], [(0, 0.8), (0.8, 0.9), (0.9, 1)]):
        LOGGER.info(f'Writing {ds} dataset to file...')
        d = copy.deepcopy(data)
        d['images'] = d['images'][int(split[0] * len(d['images'])):int(split[1] * len(d['images']))]
        LOGGER.info(f'This dataset contains: {len(d["images"])} images.')
        new_annotations = []
        for annot in d['annotations']:
            found = False
            for img in d['images']:
                if annot['image_id'] == img['id']:
                    found = True
            if found:
                new_annotations.append(annot)
        d['annotations'] = new_annotations
        with open(f'gerald_{ds}_annotations.json', 'w') as f:
            f.write(json.dumps(data))
