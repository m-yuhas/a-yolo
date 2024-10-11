from PL_DataModules.coco import COCODataModule
from PL_DataModules.voc import VOCDataModule
from PL_DataModules.coco_tensor import COCOTensorDataModule


CONFIGS = {
    'coco': COCODataModule,
    'voc': VOCDataModule,
    'coco_tensor': COCOTensorDataModule,
}


def build_data(model_name):
    return CONFIGS[model_name]
