import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from typing import Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from enum import Enum
import numpy as np
from typing import Callable, List, Any, Optional
from matplotlib import pyplot as plt
import linecache
from tqdm import tqdm

MAX_MZ = 2000
RANDOM_SEED = 43242
FINGERPRINT_NBITS = 4096
FINGERPRINT_NBITS = 512

SPECTRA_DIM = MAX_MZ * 2

def fingerprint(mol, nbits=FINGERPRINT_NBITS) -> np.array:
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=nbits)
    mol_rep = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, mol_rep)
    return mol_rep


def encode_spec(spec):
    vec = np.zeros(MAX_MZ * 2)
    for i in range(spec.shape[0]):
        mz_rnd = int(spec[i, 0])
        if mz_rnd >= MAX_MZ:
            continue
        logint= np.log10(spec[i, 1] + 1)
        if vec[mz_rnd] == 0 or vec[mz_rnd] < logint:
            vec[mz_rnd] = logint
            vec[MAX_MZ + mz_rnd] = np.log10((spec[i, 0] - mz_rnd) + 1)
    return vec

def gnps_parser(fname, from_mol: int = 0, to_mol: Optional[int] = None) -> List[Any]:
    molecules = []
    spectra = []
    with open(fname) as fl:
        for i, line in tqdm(list(enumerate(fl.readlines()[from_mol:]))):
            if to_mol is not None and from_mol + i >=  to_mol:
                break
            mol, spec = parse_spectra(line)
            molecules.append(mol)
            spectra.append(spec)
    return molecules, spectra


def parse_spectra(line):
    spec_str, smiles = line.strip().split('\t')
    spec = np.array(json.loads(spec_str))
    if not( len(spec.shape) == 2 and spec.shape[0] >= 10 and (spec <= 0).sum() == 0):
        print(spec)
        raise ValueError('what')
    # Round to 3 digits M/Z precision
    spec[:, 0] = np.round(spec[:, 0], 3)
    # We'll predict relative intensities
    #spec[:, 1] /= spec[:, 1].max()
    mol = Chem.MolFromSmiles(smiles)
    return mol, encode_spec(spec)



def decode_spec(flatspec: np.array) -> np.array:
    intensities = flatspec[:len(flatspec) // 2]
    spln = sum(intensities > 0)
    spec = np.zeros([spln, 2])
    spec[:, 1] = 10**(intensities[intensities > 0]) - 1
    spec[:, 0] = np.where(intensities > 0)[0] + (10**(flatspec[len(flatspec) // 2:][intensities > 0]) - 1)
    return spec


class Mol2PropertiesDataset(Dataset):
    SAVED_PROPS = [
            'dataset_name',
            'molecules',
            'properties',
            'mol_reps',
            'property_names'
    ]

    SAVE_DIR = 'data'

    def __init__(
        self,
        dataset_name: str,
        fname: str,
        parser: Callable,
        mol_representation: Callable = fingerprint,
        from_mol: int = 0,
        to_mol: Optional[int] = None,
        property_names: Optional[List[str]] = None,
        use_cache: bool = False,
        **mol_rep_kwargs,
    ):
        self.dataset_name = dataset_name
        if use_cache and self.is_cached():
            print('Cache found, loading dataset')
            self.load()
        else:
            self.molecules, self.properties = parser(fname, from_mol=from_mol, to_mol=to_mol)
            self.property_names = property_names
            self.mol_representation = mol_representation
            self.mol_rep_kwargs = mol_rep_kwargs

            self.mol_reps = [self.mol_representation(mol, **mol_rep_kwargs) for mol in tqdm(self.molecules)]


            if use_cache:
                print('Caching dataset')
                self.save()

    @property
    def cache_fname(self):
        return os.path.join(Mol2PropertiesDataset.SAVE_DIR, self.dataset_name + '.pkl')

    def is_cached(self):
        return os.path.exists(self.cache_fname)

    def save(self):
        with open(self.cache_fname, 'wb') as fl:
            pickle.dump({k: getattr(self, k) for k in Mol2PropertiesDataset.SAVED_PROPS}, fl)
            # Weird torch_geometric bug that needs to reload pickled object to regenerate globalstorage
            self.load()

    def load(self):
        with open(self.cache_fname, 'rb') as fl:
            props = pickle.load(fl)
            for k, v in props.items():
                setattr(self, k, v)

    def __len__(self):
        return len(self.mol_reps)

    def __getitem__(self, idx: int):
        return self.mol_reps[idx], torch.FloatTensor(self.properties[idx])


class Mol2PropertiesDiskDataset(Dataset):
    def __init__(
        self,
        fname: str,
        parser: Callable = parse_spectra,
        mol_representation: Callable = fingerprint,
        property_names: Optional[List[str]] = None,
    ):
        self.fname = fname
        self.mol_representation = mol_representation
        self.property_names = property_names
        self.parser = parser
        with open(self.fname) as fl:
            self.flen = sum(1 for _ in fl)

    def __len__(self):
        return self.flen

    def __getitem__(self, idx: int):
        line = linecache.getline(self.fname, idx)
        mol, spec = self.parser(line)
        return self.mol_representation(mol), spec



def plot_specs(flatspec1, flatspec2, labels=['one', 'two'], mol=None):
    if mol is not None:
        f = Chem.Draw.MolToMPL(mol, size=(200,200))
        plt.axis('off')
    f, axarr = plt.subplots(nrows=1,  ncols=1,figsize=(15, 3))
    s1, s2 = decode_spec(flatspec1), decode_spec(flatspec2)
    axarr.vlines(s1[:, 0], ymin=0, ymax=s1[:, 1], color='red', label=labels[0])
    axarr.vlines(s2[:, 0], ymin=-s2[:, 1], ymax=0, color='blue', label=labels[1])
    axarr.legend()
