from io import BytesIO
import os
from typing import Any, Collection

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch.utils.data.dataset import Dataset
import torch_geometric
from molxspec import utils, chemberta, graphs, models
import functools
import numpy as np
from enum import Enum
import requests
import tarfile
import shutil
from tqdm import tqdm

class ModelType(Enum):
    MLP = 'mlp'
    GCN = 'gcn'
    EGNN = 'egnn'
    BERT = 'bert'


MOL_REPS = {
    ModelType.MLP: utils.fingerprint,
    ModelType.GCN: graphs.mol_to_torch_geom,
    ModelType.BERT: chemberta.chemberta_representation_batch,
    ModelType.EGNN: functools.partial(graphs.mol_to_torch_geom, add_positions=True)
}

MODEL_CLASSES = {
    ModelType.MLP: models.Mol2SpecSimple,
    ModelType.GCN: models.Mol2SpecGraph,
    ModelType.EGNN: models.Mol2SpecEGNN,
    ModelType.BERT: models.Mol2SpecSimple,
}

class PredictionDataset(Dataset):
    def __init__(self, reps: Collection[Any]):
        self.reps = reps
    
    def __len__(self) -> int:
        return len(self.reps)
    
    def __getitem__(self, index) -> Any:
        return self.reps[index]


def __mols_from_smiles(smiless: Collection[str]) -> Collection[Chem.rdchem.Mol]:
    return [Chem.MolFromSmiles(s) for s in smiless]


def __prepare_for_egnn(mols: Chem.rdchem.Mol):
    for m in tqdm(mols, desc='Opt geom for 3D [EGNN model]'):
        AllChem.EmbedMolecule(m, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(m)


def get_model_dir() -> str:
    return os.path.join(os.path.dirname(__file__), 'models')


def get_model_path(model_type: ModelType) -> str:
    model_path = os.path.join(get_model_dir(), model_type.value + '.pt')
    return model_path


def _download_models():
    print('Downloading models...')
    url = 'https://zenodo.org/record/5717415/files/models.tgz'
    os.makedirs(get_model_dir(), exist_ok=True)
    print('Getting model files...')
    archive = BytesIO(requests.get(url))
    print('Unpacking...')
    with tarfile(fileobj=archive) as tarf:
        for mf in tarf.getmembers():
            outfname = os.path.join(get_model_dir(), mf.name.split('/')[-1])
            with open(outfname, 'w') as outf:
                shutil.copyfileobj(mf, outf)


def __load_model(model_type: ModelType):
    model_path = get_model_path(model_type)
    if not os.path.exists(model_path):
        _download_models()
    saved_state = torch.load(model_path)
    model = MODEL_CLASSES[model_type](**saved_state['model_kwargs'])
    
    model.load_state_dict(saved_state['model_state_dict'])
    for param in model.parameters():
            param.requires_grad = False
    model.eval()
    return model


def __do_prediction(reps: Collection[Any], model_type: ModelType) -> np.array:
    data = PredictionDataset(reps)
    if model_type in [ModelType.EGNN, ModelType.GCN]:
        dataloader = torch_geometric.loader.DataLoader
    else:
        dataloader = torch.utils.data.DataLoader
    loader = dataloader(data, batch_size=1024)
    model = __load_model(model_type)

    preds = []
    for inputs in loader:
        preds.append(model(inputs).numpy())
    preds = [
        utils.decode_spec(p, lowest_intensity=1e-4, make_relative=True)
        for pred in preds
        for p in pred
    ]
    return preds


def predict(smiless: Collection[str], frag_level: int, adduct: str, model_type: ModelType) -> np.array:
    mols = __mols_from_smiles(smiless)
    frag_feats = utils.get_featurized_fragmentation_level(frag_level)
    adduct_feats = utils.get_featurized_adducts(adduct)
    reps = None
    if model_type == ModelType.EGNN:
        # Molecules need to be optimized first for this model
        __prepare_for_egnn(mols)
    if model_type == ModelType.BERT:
        # For ChemBERTa, we calculate molecular representations in batch because it's quicker
        reps = MOL_REPS[model_type](mols, frag_levels=frag_feats, adduct_feats=adduct_feats)
    else:
        reps = [MOL_REPS[model_type](m, frag_levels=frag_feats, adduct_feats=adduct_feats) for m in mols]
    return __do_prediction(reps, model_type)