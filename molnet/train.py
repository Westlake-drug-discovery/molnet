import nni
import os
from parser import create_parser
import torch


if __name__ == '__main__':
    config =  create_parser()
    tuner_params = nni.get_next_parameter()
    config.update(tuner_params)
    print(config)

    from engine import Exp
    exp = Exp(config)
    print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train()
    
    print('>>>>>>>testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.model.load_state_dict(torch.load(os.path.join(exp.path, 'checkpoint.pth')))
    with torch.no_grad():
        test_loss = exp.test_epoch(exp.test_loader)
    nni.report_final_result(test_loss)