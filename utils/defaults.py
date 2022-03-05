import os
import yaml
import argparse


def argument_parser():
    parser = argparse.ArgumentParser("Joseph's lightning")
    parser.add_argument("-n", "--experiment-name", type=str)
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--resume", default=False, action="store_true", help="resume training")
    parser.add_argument("-c", "--checkpoint", default=None, type=str, help="checkpoint file")
    return parser


def load_config(cfg_filename: str):
    assert os.path.isfile(cfg_filename), f"Config file '{cfg_filename}' does not exist!"
    with open(cfg_filename, encoding='ascii', errors='ignore') as f:
        config = yaml.safe_load(f)  # model dict
    return config
