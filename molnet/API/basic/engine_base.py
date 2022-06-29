import os
import abc
import json
import random
import torch
import logging
import numpy as np
from .earlystop import EarlyStopping

def SetSeed(seed,det=True):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    if det: 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ExpBase():
    def __init__(self, config):
        self.config = config
        SetSeed(self.config["seed"])
    
    @abc.abstractmethod
    def initialization(self):
        self.path = self.config["res_dir"]+'/{}'.format(self.config['ex_name'])
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.checkpoints_path = os.path.join(self.path, 'checkpoints')
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

        sv_param = os.path.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.config, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO,
                            filename=self.path+'/log.log',
                            filemode='a',
                            format='%(asctime)s - %(message)s'
                            )
        self.epoch = self.config['epoch_s']
        self.best_val_score = np.inf
        self.early_stopping = EarlyStopping(patience=self.config['patience'], verbose=True)

        self.get_data()
        self.build_model()
        self.select_optimizer()
        self.get_loss()
    
    @abc.abstractmethod
    def build_model(self):
        return NotImplemented
    
    @abc.abstractmethod
    def get_data(self):
       return NotImplemented
    
    @abc.abstractmethod
    def get_loss(self):
        return NotImplemented

    @abc.abstractmethod
    def select_optimizer(self):
        return NotImplemented

    @abc.abstractmethod
    def _save(self, epoch):
        path = os.path.join(self.checkpoints_path, str(epoch) + '.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model_optim.state_dict(),
            'scheduler_state_dict': None if self.scheduler == None else self.scheduler.state_dict()
        }, os.path.join(path))

    @abc.abstractmethod
    def _load(self,epoch):
        path = os.path.join(self.checkpoints_path, str(epoch) + '.pth')
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler != None and checkpoint['scheduler_state_dict'] != None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch_s = checkpoint['epoch']
        self.best_val_score = checkpoint['best_val_score']

    @abc.abstractmethod
    def train_epoch(self, data_loader, model_optim=None):
        return NotImplemented

    @abc.abstractmethod
    def test_epoch(self, data_loader, model_optim=None):
        return NotImplemented
    
    @abc.abstractmethod
    def train(self, args):
        return NotImplemented