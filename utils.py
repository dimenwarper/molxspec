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
from typing import Callable, List, Any, Optional, Tuple, Dict, Collection, Union
import linecache
from tqdm import tqdm
from transformers import AutoTokenizer

MAX_MZ = 2000
RANDOM_SEED = 43242
FINGERPRINT_NBITS = 1024
CHEMBERTA_MODEL = 'seyonec/ChemBERTa-zinc-base-v1'
CHEMBERTA_MAX_LEN = 300 # Natural products aren't that large
__CHEMBERTA_TOKENIZER = None

SPECTRA_DIM = MAX_MZ * 2


def chemberta_tokenizer() -> AutoTokenizer:
    global __CHEMBERTA_TOKENIZER
    if __CHEMBERTA_TOKENIZER is None:
        __CHEMBERTA_TOKENIZER = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    return __CHEMBERTA_TOKENIZER


def fingerprint(mol, nbits=FINGERPRINT_NBITS) -> np.array:
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=nbits)
    mol_rep = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, mol_rep)
    return mol_rep


def chemberta_tokenize(mol) -> np.array:
    return chemberta_tokenizer()(
        Chem.MolToSmiles(mol), 
        return_tensors='pt',
        padding='max_length',
        max_length=CHEMBERTA_MAX_LEN,
        truncation=True
        )


def encode_spec(spec):
    vec = np.zeros(MAX_MZ * 2)
    for i in range(spec.shape[0]):
        mz_rnd = int(spec[i, 0])
        if mz_rnd >= MAX_MZ:
            continue
        logint= np.log10(spec[i, 1] + 1)
        if vec[mz_rnd] < logint:
            vec[mz_rnd] = logint
            vec[MAX_MZ + mz_rnd] = np.log10((spec[i, 0] - mz_rnd) + 1)
    return vec


def decode_spec(flatspec: np.array, lowest_intensity: float = 0) -> np.array:
    intensities = flatspec[:len(flatspec) // 2]
    spln = sum(intensities > lowest_intensity)
    spec = np.zeros([spln, 2])
    spec[:, 1] = 10**(intensities[intensities > lowest_intensity]) - 1
    spec[:, 0] = np.where(intensities > lowest_intensity)[0] + (10**(flatspec[len(flatspec) // 2:][intensities > lowest_intensity]) - 1)
    return spec


def gnps_parser(fname: str, from_mol: int = 0, to_mol: Optional[int] = None) -> List[Any]:
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


def gnps_parser_3d(fnames: Tuple[str, str], from_mol: int = 0, to_mol: Optional[int] = None) -> List[Any]:
    molecules = []
    spectra = []
    spectra_fname, sdf_fname = fnames

    for i, mol in enumerate(Chem.SDMolSupplier(sdf_fname)):
        if to_mol is not None and from_mol + i >=  to_mol:
            break
        molecules.append(mol)

    with open(spectra_fname) as fl:
        for i, line in tqdm(list(enumerate(fl.readlines()[from_mol:]))):
            if to_mol is not None and from_mol + i >=  to_mol:
                break
            _, spec = parse_spectra(line)
            spectra.append(spec)
    return molecules[:len(spectra)], spectra


def parse_spectra(line):
    spec_str, smiles = line.strip().split('\t')
    spec = np.array(json.loads(spec_str))
    if not( len(spec.shape) == 2 and spec.shape[0] >= 10 and (spec <= 0).sum() == 0):
        print(spec)
        raise ValueError('what')
    # Round to 3 digits M/Z precision
    spec[:, 0] = np.round(spec[:, 0], 3)
    # We'll predict relative intensities
    spec[:, 1] /= spec[:, 1].max()
    mol = Chem.MolFromSmiles(smiles)
    return mol, encode_spec(spec)




class Mol2PropertiesDataset(Dataset):
    SAVED_PROPS = [
            'molecules',
            'properties',
            'mol_reps',
            'property_names'
    ]

    SAVE_DIR = 'data'

    def __init__(
        self,
        dataset_name: str,
        fnames: Union[str, Collection[str]],
        parser: Callable,
        mol_representation: Callable = fingerprint,
        from_mol: int = 0,
        to_mol: Optional[int] = None,
        property_names: Optional[List[str]] = None,
        use_cache: bool = False,
        mol_rep_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        if use_cache and self.is_cached():
            print('Cache found, loading dataset')
            self.load()
        else:
            self.molecules, self.properties = parser(fnames, from_mol=from_mol, to_mol=to_mol)
            self.property_names = property_names
            self.mol_representation = mol_representation
            self.mol_rep_kwargs = mol_rep_kwargs if mol_rep_kwargs is not None else {}

            self.mol_reps = [self.mol_representation(mol, **self.mol_rep_kwargs) for mol in tqdm(self.molecules)]


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
            for k in Mol2PropertiesDataset.SAVED_PROPS:
                delattr(self, k)
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