import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
import torch, argparse
from engine.core import YAMLConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default=r'', type=str)
    args = parser.parse_args()

    model = YAMLConfig(args.config, resume=None).model

    for name, module in model.named_modules():
            print(f"Layer Name: " + name + ", Layer Type: ", module.__class__.__name__)