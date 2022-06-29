#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
In-memory dataset.
"""

import os
from os.path import join, exists

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import lmdb
import pickle

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
    """

    Parallel map using joblib.

    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.

    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1

    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
    )

    return results

def load_bace_dataset(data_path, task_names=None):
    if task_names is None:
        task_names = ['Class']

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['mol']
    labels = input_df[task_names]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans

    data_list = []
    for i in range(len(smiles_list)):
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    return data_list

def load_bbbp_dataset(data_path, task_names=None):

    if task_names is None:
        task_names = ['p_np']

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    from rdkit.Chem import AllChem
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    preprocessed_rdkit_mol_objs_list = [m if not m is None else None for m in
                                                          rdkit_mol_objs_list]
    smiles_list = [AllChem.MolToSmiles(m) if not m is None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df[task_names]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans

    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    return data_list


def load_clintox_dataset(data_path, task_names=None):

    if task_names is None:
        task_names = ['FDA_APPROVED', 'CT_TOX']

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    from rdkit.Chem import AllChem
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    preprocessed_rdkit_mol_objs_list = [m if not m is None else None 
            for m in rdkit_mol_objs_list]
    smiles_list = [AllChem.MolToSmiles(m) if not m is None else None 
            for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df[task_names]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans

    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    return data_list

def load_freesolv_dataset(data_path, task_names=None):
    if task_names is None:
        task_names = ['expt']

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    return data_list

def load_hiv_dataset(data_path, task_names=None):
    if task_names is None:
        task_names = ['HIV_active']

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans

    data_list = []
    for i in range(len(smiles_list)):
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    return data_list

def load_lipophilicity_dataset(data_path, task_names=None):
    if task_names is None:
        task_names = ['exp']

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    return data_list

def load_muv_dataset(data_path, task_names=None):
    if task_names is None:
        task_names = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
           'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
           'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]
    labels = labels.replace(0, -1)  # convert 0 to -1
    labels = labels.fillna(0)   # convert nan to 0

    data_list = []
    for i in range(len(smiles_list)):
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    return data_list

def load_qm7_dataset(data_path, task_names=None):
    """
    min/max/mean: -2192.0/-404.88/-1544.8360893118595 
    """
    if task_names is None:
        task_names = ['u0_atom']

    csv_file = join(data_path, 'raw/qm7.csv')
    input_df = pd.read_csv(csv_file, sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    return data_list

def load_qm8_dataset(data_path, task_names=None):
    """
    tbd 
    """
    if task_names is None:
        task_names = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 
            'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 
            'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM']

    csv_file = join(data_path, 'raw/qm8.csv')
    input_df = pd.read_csv(csv_file, sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    return data_list

def load_qm9_dataset(data_path, task_names=None):
    """
    tbd
    """
    if task_names is None:
        task_names = ['homo', 'lumo', 'gap']

    csv_file = join(data_path, 'raw/qm9.csv')
    input_df = pd.read_csv(csv_file, sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list = []
    for i in range(len(labels)):
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    return data_list

def load_sider_dataset(data_path, task_names=None):
    if task_names is None:
        task_names = ['Hepatobiliary disorders',
           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
           'Investigations', 'Musculoskeletal and connective tissue disorders',
           'Gastrointestinal disorders', 'Social circumstances',
           'Immune system disorders', 'Reproductive system and breast disorders',
           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
           'General disorders and administration site conditions',
           'Endocrine disorders', 'Surgical and medical procedures',
           'Vascular disorders', 'Blood and lymphatic system disorders',
           'Skin and subcutaneous tissue disorders',
           'Congenital, familial and genetic disorders',
           'Infections and infestations',
           'Respiratory, thoracic and mediastinal disorders',
           'Psychiatric disorders', 'Renal and urinary disorders',
           'Pregnancy, puerperium and perinatal conditions',
           'Ear and labyrinth disorders', 'Cardiac disorders',
           'Nervous system disorders',
           'Injury, poisoning and procedural complications']

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]
    labels = labels.replace(0, -1)  # convert 0 to -1

    data_list = []
    for i in range(len(smiles_list)):
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    return data_list

def load_tox21_dataset(data_path, task_names=None):
    if task_names is None:
        task_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]
    labels = labels.replace(0, -1)  # convert 0 to -1
    labels = labels.fillna(0)   # convert nan to 0

    data_list = []
    for i in range(len(smiles_list)):
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    return data_list

def load_toxcast_dataset(data_path, task_names=None):
    def get_default_toxcast_task_names(data_path):
        """Get that default toxcast task names and return the list of the input information"""
        raw_path = join(data_path, 'raw')
        csv_file = os.listdir(raw_path)[0]
        input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
        return list(input_df.columns)[1:]

    if task_names is None:
        task_names = get_default_toxcast_task_names(data_path)

    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    smiles_list = input_df['smiles']
    from rdkit.Chem import AllChem
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if not m is None else None 
            for m in rdkit_mol_objs_list]
    smiles_list = [AllChem.MolToSmiles(m) if not m is None else None 
            for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df[task_names]
    labels = labels.replace(0, -1)  # convert 0 to -1
    labels = labels.fillna(0)   # convert nan to 0

    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        data = {}
        data['smiles'] = smiles_list[i]        
        data['label'] = labels.values[i]
        data_list.append(data)
    return data_list


def get_dataset(dataset_name, data_path, task_names=None):
    """Return dataset according to the ``dataset_name``"""
    if dataset_name == 'bace':
        dataset = load_bace_dataset(data_path, task_names)
    elif dataset_name == 'bbbp':
        dataset = load_bbbp_dataset(data_path, task_names)
    elif dataset_name == 'clintox':
        dataset = load_clintox_dataset(data_path, task_names)
    elif dataset_name == 'hiv':
        dataset = load_hiv_dataset(data_path, task_names)
    elif dataset_name == 'muv':
        dataset = load_muv_dataset(data_path, task_names)
    elif dataset_name == 'sider':
        dataset = load_sider_dataset(data_path, task_names)
    elif dataset_name == 'tox21':
        dataset = load_tox21_dataset(data_path, task_names)
    elif dataset_name == 'toxcast':
        dataset = load_toxcast_dataset(data_path, task_names)
    else:
        raise ValueError('%s not supported' % dataset_name)

    return dataset



class ChemDataset(Dataset):
    def __init__(self, dataset_name, data_path, processed_path, memory = 10):
        super(ChemDataset, self).__init__()
        self.dataset_name = dataset_name
        self.processed_db = processed_path + '/{}.lmdb'.format(dataset_name)
        self.memory = memory
        self.db = None
        if not os.path.exists(self.processed_db):
            os.makedirs(processed_path, exist_ok=True)
            self.preprocess(dataset_name, data_path, memory)

    def preprocess(self, dataset_name, data_path, memory, drop_none=True):
        def handle_per_task(data):
            from API.basic.mol_read import ReadMol
            import rdkit.Chem as Chem

            try:
                smiles = data['smiles']
                mol = ReadMol(Chem.MolFromSmiles(smiles)).to_dict_atom()
                mol['label'] = data['label']
                mol['smiles'] = smiles
                return mol
            except:
                return None
        
        data_list = get_dataset(dataset_name, data_path)
        data_list = pmap_multi(handle_per_task, [(one,) for one in data_list])
        if drop_none:
            self.data_list = [data for data in data_list if not data is None]
        else:
            self.data_list = data_list
        
        db = lmdb.open(
            self.processed_db,
            map_size=memory*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False, 
        )

        with db.begin(write=True, buffers=True) as txn:
            for idx, data in enumerate(data_list):
                txn.put(
                            key = str(idx).encode(),
                            value = pickle.dumps(data)
                        )
        db.close()

        print("processed {} data!".format(self.dataset_name))

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_db,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)
 
    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        
        if type(idx)==list:
            data_list = []
            for i in idx:
                key = self.keys[i]
                data = pickle.loads(self.db.begin().get(key))
                data_list.append(data)
            return data_list
    
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        return data

    
