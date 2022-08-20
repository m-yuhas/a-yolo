from pytorch_lightning import Trainer, seed_everything
from utils.defaults import train_argument_parser, load_config
from utils.build_data import build_data
from utils.build_logger import build_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from PL_Modules.build_detection import build_model
from PL_Modules.pl_detection import LitDetection


def main():
    args = train_argument_parser().parse_args()

    data_cfgs = load_config(args.dataset)
    data = build_data(data_cfgs['datamodule'])
    data = data(data_cfgs)

    model_cfgs = load_config(args.model)
    model = build_model(model_cfgs)
    lightning = LitDetection(model, model_cfgs, data_cfgs)

    logger = build_logger(args.logger, model, model_cfgs, args.version)

    seed_everything(96, workers=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor='mAP',
        dirpath='logs/',
        mode="max",
        filename='{epoch:02d}-{mAP:.2f}'
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=300,
        check_val_every_n_epoch=5,
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=logger,
        callbacks=[checkpoint_callback],
        # precision=16,
        # amp_backend="apex",
        # amp_level=01,
        # auto_lr_find=True,
        # benchmark=False,
        # default_root_dir="lightning_logs",
        # detect_anomaly=True,
        # limit_train_batches=80,
        # limit_val_batches=40,
        # reload_dataloaders_every_n_epochs=10,
    )

    trainer.fit(lightning, datamodule=data, ckpt_path='logs/epoch=29-mAP=0.22.ckpt')
    # trainer.tune(lightning, datamodule=data)
    # trainer.validate(lightning, datamodule=data, ckpt_path='')
    # trainer.test(lightning, datamodule=data)


if __name__ == "__main__":
    main()
