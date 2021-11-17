import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from typing import Optional
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import numpy as np
from typing import Callable, List, Any, Optional, Tuple, Dict, Collection, Union
from tqdm import tqdm

MAX_MZ = 2000
ADDUCTS = ['[M+H]+', '[M+Na]+', 'M+H', 'M-H', '[M-H2O+H]+', '[M-H]-', '[M+NH4]+', 'M+NH4', 'M+Na']
RANDOM_SEED = 43242
FINGERPRINT_NBITS = 1024

MAX_ION_SHIFT = 25
FRAGMENT_LEVELS = [-4, -3, -2, -1, 0]
SPECTRA_DIM = MAX_MZ * 2


def get_fragmentation_level(mol, spec):
    mass = ExactMolWt(mol)
    min_mass, max_mass = int(max(0, mass - MAX_ION_SHIFT)), int(min(mass + MAX_ION_SHIFT, MAX_MZ))
    _spec = 10 ** spec - 1
    frag_level = max(0.01, _spec[min_mass:max_mass + 1].sum())
    return np.log10(frag_level / _spec.sum())


def get_featurized_adducts(adduct):
    return np.array([int(adduct == ADDUCTS[i]) for i in range(len(ADDUCTS))])


def get_featurized_fragmentation_level(mol, spec):
    fl = get_fragmentation_level(mol, spec)
    flf = [int(fl <= FRAGMENT_LEVELS[0])]
    flf += [int(FRAGMENT_LEVELS[i-1] <= fl <= FRAGMENT_LEVELS[i]) for i in range(1, len(FRAGMENT_LEVELS))]
    return np.array(flf)


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


def fingerprint(mol, frag_levels, adduct_feats, nbits=FINGERPRINT_NBITS) -> np.array:
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=nbits)
    mol_rep = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, mol_rep)
    return np.hstack([mol_rep, frag_levels, adduct_feats])


def gnps_parser(fname: str, from_mol: int = 0, to_mol: Optional[int] = None) -> List[Any]:
    molecules = []
    spectra = []
    adducts = []
    with open(fname) as fl:
        for i, line in tqdm(list(enumerate(fl.readlines()[from_mol:]))):
            if to_mol is not None and from_mol + i >=  to_mol:
                break
            _, adduct, mol, spec = parse_spectra(line)
            molecules.append(mol)
            spectra.append(spec)
            adducts.append(adduct)
    return molecules, adducts, spectra


def gnps_parser_3d(fnames: Tuple[str, str], from_mol: int = 0, to_mol: Optional[int] = None) -> List[Any]:
    molecules = []
    spectra = []
    adducts = []
    spectra_fname, sdf_fname = fnames

    for i, mol in enumerate(Chem.SDMolSupplier(sdf_fname)):
        if to_mol is not None and from_mol + i >=  to_mol:
            break
        molecules.append(mol)

    with open(spectra_fname) as fl:
        for i, line in tqdm(list(enumerate(fl.readlines()[from_mol:]))):
            if to_mol is not None and from_mol + i >=  to_mol:
                break
            _, adduct, _, spec = parse_spectra(line)
            adducts.append(adduct)
            spectra.append(spec)
    return molecules[:len(spectra)], adducts, spectra


def parse_spectra(line):
    sid, adduct, spec_str, smiles = line.strip().split('\t')
    spec = np.array(json.loads(spec_str))
    if not( len(spec.shape) == 2 and spec.shape[0] >= 10 and (spec <= 0).sum() == 0):
        print(spec)
        raise ValueError('what')
    # Round to 3 digits M/Z precision
    spec[:, 0] = np.round(spec[:, 0], 3)
    # We'll predict relative intensities
    spec[:, 1] /= spec[:, 1].max()
    mol = Chem.MolFromSmiles(smiles)
    return sid, adduct, mol, encode_spec(spec)


class Mol2SpecDataset(Dataset):
    SAVED_PROPS = [
            'molecules',
            'spectra',
            'frag_levels',
            'adducts',
            'adduct_feats',
            'mol_reps',
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
        use_cache: bool = False,
        mol_rep_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        if use_cache and self.is_cached():
            print('Cache found, loading dataset')
            self.load()
        else:
            self.molecules, self.adducts, self.spectra = parser(fnames, from_mol=from_mol, to_mol=to_mol)
            self.frag_levels = [get_featurized_fragmentation_level(mol, spec) for mol, spec in zip(self.molecules, self.spectra)]
            self.adduct_feats = [get_featurized_adducts(adduct) for adduct in self.adducts]
            self.mol_representation = mol_representation
            self.mol_rep_kwargs = mol_rep_kwargs if mol_rep_kwargs is not None else {}

            self.mol_reps = [
                self.mol_representation(mol, frag_levels=self.frag_levels[i], adduct_feats=self.adduct_feats[i], **self.mol_rep_kwargs) 
                for i, mol in tqdm(list(enumerate(self.molecules)), desc='Calculating mol reps')
                ]

            if use_cache:
                print('Caching dataset')
                self.save()

    @property
    def cache_fname(self):
        return os.path.join(Mol2SpecDataset.SAVE_DIR, self.dataset_name + '.pkl')

    def is_cached(self):
        return os.path.exists(self.cache_fname)

    def save(self):
        with open(self.cache_fname, 'wb') as fl:
            pickle.dump({k: getattr(self, k) for k in Mol2SpecDataset.SAVED_PROPS}, fl)
            # Weird torch_geometric bug that needs to reload pickled object to regenerate globalstorage
            for k in Mol2SpecDataset.SAVED_PROPS:
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
        return self.mol_reps[idx], torch.FloatTensor(self.spectra[idx])
