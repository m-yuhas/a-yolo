from pytorch_lightning import Trainer, seed_everything
from utils.defaults import train_argument_parser, load_config
from utils.build_data import build_data
from utils.build_logger import build_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from PL_Modules.build_detection import build_model
#from PL_Modules.build_multiexit import build_model
from PL_Modules.pl_detection import LitDetection


import torch #MJY Jan 3, needed for saving model at train end

def main():
    args = train_argument_parser().parse_args()

    data_cfgs = load_config(args.dataset)
    data = build_data(data_cfgs['datamodule'])
    data = data(data_cfgs)
    model_cfgs = load_config(args.model)
    model = build_model(model_cfgs, data_cfgs['num_classes'])

    logger = build_logger(args.logger, data_cfgs['name'], args.experiment_name, model, model_cfgs)

    seed_everything(96, workers=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='mAP',
        mode="max",
        filename='{epoch:02d}-{mAP:.3f}',
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
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
    if not args.test:
        lightning = LitDetection(model, model_cfgs, data_cfgs)
        trainer.fit(lightning, datamodule=data)
        torch.save(lightning.model, f'{args.experiment_name}.pt') #MJY Jan 3 save model after train
    else:
        test_cfgs = {'visualize': args.visualize, 'test_nms': args.nms, 'test_conf': args.conf,
                     'show_dir': args.show_dir, 'show_score_thr': args.show_score_thr}
        model = torch.load(args.ckpt, map_location=torch.device('cpu'))
        #model.to(torch.device('cpu'))
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            enable_progress_bar=True,
            logger=logger,
            callbacks=[checkpoint_callback],
            detect_anomaly=True,
        )

        lightning = LitDetection(model, model_cfgs, data_cfgs, test_cfgs)
        lightning.to(torch.device('cpu'))
        #trainer.test(lightning, datamodule=data) #,
        trainer.validate(lightning, datamodule=data)
    # trainer.tune(lightning, datamodule=data)
    # trainer.validate(lightning, datamodule=data, ckpt_path='weights/al6/epoch=399-mAP=0.774.ckpt')


if __name__ == "__main__":
    main()
