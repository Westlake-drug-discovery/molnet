import numpy as np
from rdkit import Chem
from .mol_kit import CompoundKit, Compound3DKit

class ReadMol(object):

    def __init__(self, data, mode='auto'):
        super().__init__()
        self.data = data
        if type(data) == Chem.rdchem.Mol:
            self.mol = data
        elif data[-5:].lower() == '.mol2':
            self.mol = Chem.MolFromMol2File(data, sanitize=False)
        else:
            if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
                with open(data, 'r') as f:
                    self.block = f.read()
            else:
                self.block = data
            self.mol = Chem.rdmolfiles.MolFromPDBBlock(self.block)
        
        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.charge = []

    def mol_to_graph(self, mol):
        if len(mol.GetAtoms()) == 0:
            return None

        atom_id_names = [
            "atomic_num", "chiral_tag", "degree", "explicit_valence", 
            "formal_charge", "hybridization", "implicit_valence", 
            "is_aromatic", "total_numHs", "num_radical_e","valence_out_shell",
            "van_der_waals_radis"
        ]
        bond_id_names = [
            "bond_dir", "bond_type", "is_in_ring",
            "is_conjugated", "bond_stereo"
        ]
        
        data = {}
        for name in atom_id_names:
            data[name] = []
        data['mass'] = []
        for name in bond_id_names:
            data[name] = []
        data['edges'] = []

        ### atom features
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return None
            for name in atom_id_names:
                data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
            data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

        ### bond features
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # i->j and j->i
            data['edges'] += [(i, j), (j, i)]
            for name in bond_id_names:
                bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1   # 0: OOV
                data[name] += [bond_feature_id] * 2

        ### self loop (+2)
        N = len(data[atom_id_names[0]])
        for i in range(N):
            data['edges'] += [(i, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2   # N + 2: self loop
            data[name] += [bond_feature_id] * N

        ### check whether edge exists
        if len(data['edges']) == 0: # mol has no bonds
            for name in bond_id_names:
                data[name] = np.zeros((0,), dtype="int64")
            data['edges'] = np.zeros((0, 2), dtype="int64")

        ### make ndarray and check length
        for name in atom_id_names:
            data[name] = np.array(data[name], 'int64')
        data['mass'] = np.array(data['mass'], 'float32')
        for name in bond_id_names:
            data[name] = np.array(data[name], 'int64')
        data['edges'] = np.array(data['edges'], 'int64')

        ### morgan fingerprint
        data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
        # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
        data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
        data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
        return data

    def mol_to_3dgraph(self, mol):
        if len(mol.GetAtoms()) == 0:
            return None

        if len(mol.GetAtoms()) <= 400:
            mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
        else:
            atom_poses = Compound3DKit.get_2d_atom_poses(mol)

        data = self.mol_to_graph(mol)

        data['atom_pos'] = np.array(atom_poses, 'float32')
        data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
        BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
                Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
        data['BondAngleGraph_edges'] = BondAngleGraph_edges
        data['bond_angle'] = np.array(bond_angles, 'float32')
        return data

    def to_dict_atom(self, rdkit_pos=True): # 原子级别的特征
        if rdkit_pos:
            data = self.mol_to_3dgraph(self.mol)
            return data
        else:
            conf = self.mol.GetConformer()
            lig_coords = conf.GetPositions()
            data = self.mol_to_3dgraph(self.mol)
            data['atom_pos'] = lig_coords
            return data