from distutils.command.config import config
from unittest import result
from API.basic.engine_base import ExpBase
import logging
import nni
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class Exp(ExpBase):
    def __init__(self, config):
        super(Exp, self).__init__(config)
        self.device = torch.device('cuda:{}'.format(0))
        
        if config['model']=='GIN':
            from configs.GIN_conf import model_config
            self.enc_config = model_config
        
        from configs.downstream_conf import model_config
        self.downstream_config = model_config

        self.initialization()


    def build_model(self):
        from models.GIN import GINNet
        from configs.feat_conf import atom_names, bond_names, bond_float_names
        model_config = self.enc_config
        self.encoder = GINNet(
                        atom_names, bond_names, bond_float_names,
                        model_config['layer_num'], 
                        model_config['embed_dim'], 
                        model_config['droupout_rate'], 
                        model_config['pool']).to(self.device)
        
        from models.downstream import DownstreamModel
        self.model = DownstreamModel(self.encoder, self.config['dataname'], **self.downstream_config).to(self.device)
    
    def get_data(self):
        from API.datasets.chem_dataset import ChemDataset
        config = self.config
        dataset = ChemDataset(
                    config['dataname'], 
                    config['dataroot']+'/'+config['dataname'], 
                    config['processed_path'])

        from API.datasets.splitter import create_splitter
        splitter = create_splitter('scaffold')
        train_dataset, valid_dataset, test_dataset = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("train:{} \tvalid:{} \ttest:{}".format(len(train_dataset)), len(valid_dataset), len(test_dataset))
        
        from torch.utils.data import DataLoader
        from torch_geometric.data import Data
        from torch_geometric.data import Batch

        def graph_collate(batch):
            batch1 = [Data(
                          x = torch.tensor(one['atomic_num']),
                          atomic_num = torch.tensor(one['atomic_num']),
                          chiral_tag = torch.tensor(one['chiral_tag']),
                          degree = torch.tensor(one['degree']),
                          explicit_valence = torch.tensor(one['explicit_valence']),
                          formal_charge = torch.tensor(one['formal_charge']),
                          hybridization = torch.tensor(one['hybridization']),
                          implicit_valence = torch.tensor(one['implicit_valence']),
                          is_aromatic = torch.tensor(one['is_aromatic']),
                          total_numHs = torch.tensor(one['total_numHs']),
                          mass = torch.tensor(one['mass']).reshape(-1,1),
                          bond_dir = torch.tensor(one['bond_dir']),
                          bond_type = torch.tensor(one['bond_type']),
                          is_in_ring = torch.tensor(one['is_in_ring']),
                          num_radical_e = torch.tensor(one['num_radical_e']),
                          valence_out_shell = torch.tensor(one['valence_out_shell']).reshape(-1),
                          van_der_waals_radis = torch.tensor(one['van_der_waals_radis']).reshape(-1),
                          edge_index = torch.tensor(one['edges']).T,
                          morgan_fp = torch.tensor(one['morgan_fp']).reshape(1,-1),
                          daylight_fg_counts = torch.tensor(one['daylight_fg_counts']).reshape(1,-1),
                          atom_pos = torch.tensor(one['atom_pos']),
                          bond_length = torch.tensor(one['bond_length']),
                          is_conjugated = torch.tensor(one['is_conjugated']),
                          bond_stereo = torch.tensor(one['bond_stereo']),
                          BondAngleGraph_edges = torch.tensor(one['BondAngleGraph_edges']).T,
                          bond_angle = torch.tensor(one['bond_angle']),
                          label = torch.tensor(one['label']).float().reshape(1,-1),
                          smiles = one['smiles']
            ) for one in batch]

            batch1 = Batch.from_data_list(batch1, follow_batch=["atomic_num","bond_length"], exclude_keys=['BondAngleGraph_edges'])

            batch2 = [Data(
                          x = torch.tensor(one['bond_length']),
                          edge_index = torch.tensor(one['BondAngleGraph_edges']).T,
                          bond_angle = torch.tensor(one['bond_angle'])
            ) for one in batch]
            batch2 = Batch.from_data_list(batch2, follow_batch=['bond_length'])
            batch1.BondAngleGraph_edges = batch2.edge_index
            batch1.bond_angle = batch2.bond_angle
            batch1.bond_length_batch = batch2.batch
            return batch1

        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=graph_collate,pin_memory=True, num_workers=8)

        self.val_loader = DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=graph_collate,pin_memory=True, num_workers=8) 

        self.test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=graph_collate, pin_memory=True, num_workers=8)

    def get_loss(self):
        self.criterion = nn.BCELoss(reduction='none')

    def select_optimizer(self):
        config = self.config
        self.model_optim = torch.optim.Adam([{"params": self.model.parameters()}], lr=config['lr'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.model_optim, step_size=30, gamma=0.8)
    
    def grad_filter(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = torch.nan_to_num(p.grad, 0.0,0.0,0.0)
    
    def calc_rocauc_score(self, labels, preds, valid):
        """compute ROC-AUC and averaged across tasks"""
        from sklearn.metrics import roc_auc_score
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
            preds = preds.reshape(-1, 1)

        pos_samples = []
        rocauc_list = []
        for i in range(labels.shape[1]):
            c_valid = valid[:, i].astype("bool")
            c_label, c_pred = labels[c_valid, i], preds[c_valid, i]
            #AUC is only defined when there is at least one positive data.
            if len(np.unique(c_label)) == 2:
                rocauc_list.append(roc_auc_score(c_label, c_pred))
                pos_samples.append(c_label.sum())

        print('Valid ratio: %s' % (np.mean(valid)))
        print('Task evaluated: %s/%s' % (len(rocauc_list), labels.shape[1]))
        if len(rocauc_list) == 0:
            raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")
        return sum(rocauc_list)/len(rocauc_list)

    def train_epoch(self, data_loader, model_optim=None):
        self.model.train()
        config = self.config
        y_scores = []
        y_true = []
        pbar = tqdm(data_loader)
        for i, batch in enumerate(pbar):
            if model_optim!=None:
                model_optim.zero_grad()
            
            batch = batch.to(self.device)
            y = (batch.label.float()+1)/2
            pred = self.model(batch)
            y_scores.append(pred)
            y_true.append(y)

            if config['dataname'] not in ["freesolv", "esol", "lipophilicity", "qm7", "qm8", "qm9"]:
                is_valid = y!=0.5
                loss_mat = self.criterion(pred, y)
                loss = torch.sum(loss_mat*is_valid)/torch.sum(is_valid)
            else:
                y = torch.tensor(batch.label).view(pred.shape).to(torch.float64).to(self.device)
                loss = self.criterion(pred, y)
            
            pbar.set_description('train loss: {:.4f}'.format(loss.item()))
            
            if model_optim!=None:
                loss.backward()
                self.grad_filter(self.model)
                model_optim.step()
                model_optim.zero_grad()

        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).detach().cpu().numpy()
        if config['dataname'] not in ["freesolv", "esol", "lipophilicity", "qm7", "qm8", "qm9"]:
            is_valid = y_true!=0.5
            result = -self.calc_rocauc_score(y_true, y_scores, is_valid)
        else:
            pass
        return result
    
    @torch.no_grad()
    def eval_epoch(self, data_loader, model_optim=None):
        self.model.eval()
        config = self.config
        y_scores = []
        y_true = []
        pbar = tqdm(data_loader)
        for i, batch in enumerate(pbar):
            batch = batch.to(self.device)
            y = (batch.label.float()+1)/2
            pred = self.model(batch)
            y_scores.append(pred)
            y_true.append(y)

            if config['dataname'] not in ["freesolv", "esol", "lipophilicity", "qm7", "qm8", "qm9"]:
                is_valid = y!=0.5
                loss_mat = self.criterion(pred, y)
                loss = torch.sum(loss_mat*is_valid)/torch.sum(is_valid)
            else:
                y = torch.tensor(batch.label).view(pred.shape).to(torch.float64).to(self.device)
                loss = self.criterion(pred, y)
            
            pbar.set_description('train loss: {:.4f}'.format(loss.item()))
            
            if model_optim!=None:
                loss.backward()
                self.grad_filter(self.model)
                model_optim.step()
                model_optim.zero_grad()

        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).detach().cpu().numpy()
        if config['dataname'] not in ["freesolv", "esol", "lipophilicity", "qm7", "qm8", "qm9"]:
            is_valid = y_true!=0.5
            result = -self.calc_rocauc_score(y_true, y_scores, is_valid)
        else:
            pass
        return result
    
    def train(self):
        config = self.config
        for epoch in range(config['epoch_s'], config['epoch_e']):
            print('start training, epoch {}'.format(epoch))
            logging.info('start training, epoch {}'.format(epoch))
            acc_train = self.train_epoch(self.train_loader, self.model_optim)

            acc_val = self.eval_epoch(self.val_loader)
            acc_test = self.eval_epoch(self.test_loader)
            print('epoch {} \t acc_train {} \t acc_val {} \t acc_test {}\n'.format(epoch, acc_train, acc_val, acc_test))
            logging.info('epoch {} \t acc_train {} \t acc_val {} \t acc_test {}\n'.format(epoch, acc_train, acc_val, acc_test))
            nni.report_intermediate_result(acc_test)
            if self.scheduler != None:
                self.scheduler.step()
            
            if acc_val<self.best_val_score:
                self.best_val_score = acc_val

            self.early_stopping(acc_val, self.model, self.path)
            if self.early_stopping.early_stop:
                print("Early stopping")
                logging.info("Early stopping")
                break
            torch.cuda.empty_cache()
            
        best_model_path = self.path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model