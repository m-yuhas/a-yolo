import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.data.datasets.cocoDataset import COCODataset
from models.data.mosaic_detection import MosaicDetection
from models.data.augmentation.data_augments import TrainTransform, ValTransform
from torch.utils.data.sampler import BatchSampler, RandomSampler


class ALCOCODataModule(pl.LightningDataModule):
    def __init__(self, cfgs):
        super().__init__()
        self.dataset_train = None
        self.dataset_val = None
        self.cd = cfgs['dataset']
        self.ct = cfgs['transform']
        # dataloader parameters
        self.data_dir = self.cd['dir']
        self.train_dir = self.cd['train']
        self.train_json_path = self.cd['train_json']
        self.val_dir = self.cd['val']
        self.val_json_path = self.cd['val_json']
        self.img_size_train = tuple(self.cd['train_size'])
        self.img_size_val = tuple(self.cd['val_size'])
        self.train_batch_size = self.cd['train_batch_size']
        self.val_batch_size = self.cd['val_batch_size']
        # transform parameters
        self.hsv_prob = self.ct['hsv_prob']
        self.flip_prob = self.ct['flip_prob']
        # mosaic
        self.mosaic_prob = self.ct['mosaic_prob']
        self.mosaic_scale = self.ct['mosaic_scale']
        self.degrees = self.ct['degrees']
        self.translate = self.ct['translate']
        self.shear = self.ct['shear']
        self.perspective = self.ct['perspective']
        # copypaste
        self.copypaste_prob = self.ct['copypaste_prob']
        self.copypaste_scale = self.ct['copypaste_scale']
        # cutpaste
        self.cutpaste_prob = self.ct['cutpaste_prob']
        # cutout rounding background
        self.cutoutR_prob = self.ct['cutoutR_prob']

    def train_dataloader(self):
        self.dataset_train = COCODataset(
            self.data_dir,
            name=self.train_dir,
            json=self.train_json_path,
            img_size=self.img_size_train,
            preprocess=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=True
        )
        self.dataset_train = MosaicDetection(
            self.dataset_train,
            mosaic_prob=self.mosaic_prob,
            mosaic_scale=self.mosaic_scale,
            img_size=self.img_size_train,
            preprocess=TrainTransform(
                max_labels=100,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,),
            degrees=self.degrees,
            translate=self.translate,
            shear=self.shear,
            perspective=self.perspective,
            copypaste_prob=self.copypaste_prob,
            copypaste_scale=self.copypaste_scale,
            cutpaste_prob=self.cutpaste_prob,
            cutoutR_prob=self.cutoutR_prob
        )
        sampler = RandomSampler(self.dataset_train)
        batch_sampler = BatchSampler(sampler, batch_size=self.train_batch_size, drop_last=False)
        train_loader = DataLoader(self.dataset_train, batch_sampler=batch_sampler,
                                  num_workers=6, pin_memory=False,  persistent_workers=True)
        return train_loader

    def val_dataloader(self):
        self.dataset_val = COCODataset(
            self.data_dir,
            name=self.val_dir,
            json=self.val_json_path,
            img_size=self.img_size_val,
            preprocess=ValTransform(legacy=False),
            cache=True,
        )
        sampler = torch.utils.data.SequentialSampler(self.dataset_val)
        val_loader = DataLoader(self.dataset_val, batch_size=self.val_batch_size, sampler=sampler,
                                num_workers=6, pin_memory=False, shuffle=False,  persistent_workers=True)
        return val_loader
