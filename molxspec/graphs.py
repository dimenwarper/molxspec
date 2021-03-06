'''
Stiched and forked from some of deepchem and https://www.kaggle.com/hd2019/using-rdkit-and-pyg-for-graph-model-dataset
'''
import os
import torch
import torch_geometric
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import numpy as np
import networkx as nx


ATOM_TYPES = [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        ]

# Number of features is one hot per atom type + unknown
# plus other features (aromatic, acceptor, etc, see node_features)
NUM_NODE_FEATURES = 108

# 4 Bond types, see edge_features
NUM_EDGE_FEATURES = 4

fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
CHEM_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef_name)

def node_features(g: nx.DiGraph) -> torch.Tensor:
    feat = []
    for n, d in g.nodes(data=True):
        h_t = []
        # Atom type (One-hot) 
        h_t += [int(d['a_type'] == x) for x in ATOM_TYPES] + [1 if d['a_type'] not in ATOM_TYPES else 0]
        # Atomic number
        h_t += [int(d['a_num'] == x) for x in 1 + np.arange(30)] + [1 if d['a_num'] > 30 else 0]
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['aromatic']))
        # Hybradization
        h_t += [int(d['hybridization'] == x) \
                for x in (Chem.rdchem.HybridizationType.SP, \
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3)]
        # Num hydrogens
        h_t += [int(d['num_h'] == x) for x in np.arange(5)] + [1 if d['num_h'] > 4 else 0]
        # Charge
        h_t.append(int(d['formal_charge']))
        h_t.append(int(d['num_rad_ele']))
        # Degree
        h_t += [int(d['deg'] == x) for x in 1 + np.arange(10)] + [1 if d['deg'] > 10 else 0]
        # Implicit Valence
        h_t += [int(d['imp_valence'] == x) for x in np.arange(7)] + [1 if d['imp_valence'] > 6 else 0]
        feat.append((n, h_t))
    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])
    return node_attr


def edge_features(g: nx.DiGraph) -> torch.Tensor:
    e={}
    for n1, n2, d in g.edges(data=True):
        e_t = [int(d['b_type'] == x)
                for x in (Chem.rdchem.BondType.SINGLE, \
                        Chem.rdchem.BondType.DOUBLE, \
                        Chem.rdchem.BondType.TRIPLE, \
                        Chem.rdchem.BondType.AROMATIC)]
        e[(n1, n2)] = e_t

    if len(e) == 0:
        n = g.nodes[0]
        e[(0, 0)] = [0, 0, 0, 0]
    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))
    return edge_index, edge_attr



def mol_to_nx(mol: Chem.rdchem.Mol) -> nx.DiGraph:
    feats = CHEM_FEATURE_FACTORY.GetFeaturesForMol(mol)

    g = nx.DiGraph()
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        g.add_node(
            i, 
            a_type=atom_i.GetSymbol(), 
            a_num=atom_i.GetAtomicNum(), 
            acceptor=0, 
            donor=0,
            aromatic=atom_i.GetIsAromatic(), 
            hybridization=atom_i.GetHybridization(),
            num_h=atom_i.GetTotalNumHs(),
            imp_valence=atom_i.GetImplicitValence(),
            formal_charge=atom_i.GetFormalCharge(),
            num_rad_ele=atom_i.GetNumRadicalElectrons(),
            deg=atom_i.GetDegree(),
            )

    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Donor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.nodes[i]['donor'] = 1
        elif feats[i].GetFamily() == 'Acceptor':
            node_list = feats[i].GetAtomIds()
            for i in node_list:
                g.nodes[i]['acceptor'] = 1
    # Read Edges
    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(i, j, b_type=e_ij.GetBondType())
    return g


def mol_to_torch_geom(mol, add_positions: bool = False, **kwargs) -> torch_geometric.data.Data:
    g = mol_to_nx(mol)
    node_attr = node_features(g)
    edge_index, edge_attr = edge_features(g)
    if add_positions:
        conformer = list(mol.GetConformers())[0]
        pos = torch.FloatTensor(conformer.GetPositions())
    else:
        pos = None
    kwargs = {k: torch.FloatTensor(v) for k, v in kwargs.items()}
    data = Data(
            x=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            **kwargs
            )
    return data
