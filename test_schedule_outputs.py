from pytorch_lightning import Trainer, seed_everything
from utils.defaults import train_argument_parser, load_config
from utils.build_data import build_data
from utils.build_logger import build_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from PL_Modules.build_detection import build_model
#from PL_Modules.build_multiexit import build_model
from PL_Modules.pl_detection import LitDetection


import torch #MJY Jan 3, needed for saving model at train end
import pytorch_lightning


class DummyDetector(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.backbone = None
        self.neck = None
        self.head = None
        self.loss = None

    def forward(self, x):
        return x


def main():
    args = train_argument_parser().parse_args()

    data_cfgs = load_config(args.dataset)
    data = build_data(data_cfgs['datamodule'])
    data = data(data_cfgs)
    model = DummyDetector()
    #logger = build_logger(args.logger, data_cfgs['name'], args.experiment_name, model, model_cfgs)

    seed_everything(96, workers=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='mAP',
        mode="max",
        filename='{epoch:02d}-{mAP:.3f}',
    )

    if not args.test:
        raise Exception("This script is for testing only! Please rerun with --test selected.")
    else:
        test_cfgs = {'visualize': args.visualize, 'test_nms': args.nms, 'test_conf': args.conf,
                     'show_dir': args.show_dir, 'show_score_thr': args.show_score_thr}
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            enable_progress_bar=True,
            #logger=logger,
            callbacks=[checkpoint_callback],
            detect_anomaly=True,
        )

        model_cfgs = {
            'optimizer': {
                'name': "SGD",
                'learning_rate': 0.0001,
                'momentum': 0.9,
                'weight_decay': 0.0005,
                'lr_scheduler': "CosineWarmupScheduler",
                'warmup': 0.1,
                'ema': False,
            }
        }
        lightning = LitDetection(model, model_cfgs, data_cfgs, test_cfgs)
        lightning.to(torch.device('cpu'))
        #trainer.test(lightning, datamodule=data) #,
        trainer.validate(lightning, datamodule=data)


if __name__ == "__main__":
    main()
