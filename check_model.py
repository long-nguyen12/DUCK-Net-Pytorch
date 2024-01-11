import torch 
import argparse
import yaml
from semseg.models import *
from semseg.datasets.colondb import * 
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info


def main(cfg):
    device = torch.device(cfg['DEVICE'])
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], 2)
    model.init_pretrained(model_cfg['PRETRAINED'])
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    input_tensor = torch.randn(1, 3, 352, 352).to(device)

    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")

    flops, params = get_model_complexity_info(model, (3, 352, 352), as_strings=True, print_per_layer_stat=True)

    print('FLOPs:', flops)

    print('Parameters:', params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/resnet_custom.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    
    main(cfg)
    cleanup_ddp()