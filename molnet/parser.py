import json
import argparse

# 需要修改的参数
def create_parser():
    """Creates a parser with all the variables that can be edited by the user.

    Returns:
        parser: a parser for the command line
    """
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--res_dir',default='/gaozhangyang/experiments/molnet/molnet/results',type=str)
    parser.add_argument('--ex_name', default='debug', type=str) 
    parser.add_argument('--gpu', default=2, type=int)

    # dataset parameters
    parser.add_argument('--dataroot',default="/gaozhangyang/experiments/molnet/data/chemrl_downstream_datasets")
    parser.add_argument('--dataname', default="toxcast", choices=['bbbp', 'toxcast', 'tox21', 'sider', 'esol', 'bace', 'freesolv', 'clintox', 'lipophilicity', 'hiv', 'muv', 'pcba', 'qm7', 'qm8', 'qm9'])
    parser.add_argument('--processed_path',default="/gaozhangyang/experiments/molnet/data/processed")
    parser.add_argument('--batch_size',default=128,type=int,help='Batch size')
    parser.add_argument('--seed', type=int, default=1, help='seed for reproducibility')
    
    # Training parameters
    parser.add_argument('--epoch_s', default=0, type=int, help='start epoch')
    parser.add_argument('--epoch_e', default=150, type=int, help='end epoch')
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--patience', default=100,type=int)
    parser.add_argument('--lr',default=0.01,type=float,help='Learning rate') 
    parser.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    parser.add_argument('--lr_scheduler', default='StepLR' ,type=str, help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR, StepLR,OneCycleLR]')
    
    # Model parameters
    parser.add_argument('--model', default="GIN", type=str, choices=['GIN'])
    args = parser.parse_args()
    config = args.__dict__
    return config