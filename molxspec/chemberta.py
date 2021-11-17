
from rdkit import Chem
from transformers import AutoTokenizer
from transformers import AutoModelWithLMHead
import numpy as np

CHEMBERTA_MODEL_PATH = 'seyonec/ChemBERTa-zinc-base-v1'
CHEMBERTA_MAX_LEN = 300 # Natural products aren't that large strings...maybe
__CHEMBERTA_TOKENIZER = None
__CHEMBERTA_MODEL = None


def chemberta_tokenizer() -> AutoTokenizer:
    global __CHEMBERTA_TOKENIZER
    if __CHEMBERTA_TOKENIZER is None:
        __CHEMBERTA_TOKENIZER = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_PATH)
    return __CHEMBERTA_TOKENIZER


def get_model() -> AutoModelWithLMHead:
    global __CHEMBERTA_MODEL
    if __CHEMBERTA_MODEL is None:
        print('Loading ChemBERTa')
        __CHEMBERTA_MODEL = AutoModelWithLMHead.from_pretrained(CHEMBERTA_MODEL_PATH)
        for param in __CHEMBERTA_MODEL.parameters():
            param.requires_grad = False
    return __CHEMBERTA_MODEL


def get_model_dim() -> int:
    return get_model().lm_head.dense.in_features


def chemberta_tokenize(mol) -> np.array:
    return chemberta_tokenizer()(
        Chem.MolToSmiles(mol),
        return_tensors='pt',
        padding='max_length',
        max_length=CHEMBERTA_MAX_LEN,
        truncation=True
        )


def chemberta_representation(mol, frag_levels, adduct_feats) -> np.array:
    tok = chemberta_tokenize(mol)
    model = get_model()
    x = {k: v.squeeze(dim=1) for k, v in tok.items()}
    mol_rep = model (**x, output_hidden_states=True).hidden_states[-1].mean(axis=1).numpy()
    return np.hstack([mol_rep.ravel(), frag_levels, adduct_feats])