import utils
import models
from training_setup import TrainingSetup
import torch
import models
from torch import optim
from tqdm import tqdm


def load_dataset():
    print('Loading dataset...')

    return utils.Mol2PropertiesDataset(
            'bert_gnps',
            'data/pos_processed_gnps_shuffled_with_3d_train.tsv',
            parser=utils.gnps_parser,
            mol_representation=utils.chemberta_tokenize,
            #from_mol=0,
            #to_mol=10,
            )


def load_models(hparams):
    print('Loading models...')
    _models = {}
    for hdim in hparams['hdim']:
        for n_layers  in hparams['n_layers']:
            _models[f'bert_hdim_{hdim}_layers_{n_layers}'] = models.Mol2SpecBERT(
                    prop_dim=utils.SPECTRA_DIM,
                    hdim=hdim,
                    n_layers=n_layers
                    )
    return _models


def main():
    hparams = {
            'hdim': [512, 1024, 2048],
            'batch_size': [16, 32],
            'n_layers': [3, 5]
            }
    dataset = load_dataset()
    _models = load_models(hparams)


    setups = {}
    for bsz in hparams['batch_size']:
        for mname, model in _models.items():
            setup_name = f'model_{mname}_bs_{bsz}_adam'
            setups[setup_name] = TrainingSetup(
                    model=model,
                    dataset=dataset,
                    outdir=f'runs/{setup_name}',
                    batch_size=bsz,
                    n_epochs=50,
                    optimizer=optim.Adam,
                    dataloader=torch.utils.data.DataLoader
                    )

    pbar = tqdm(list(setups.items()))
    for name, tsetup in pbar:
        pbar.set_description(f'Tng: {name}')
        tsetup.train()


if __name__ == '__main__':
    main()
