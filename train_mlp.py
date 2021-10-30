import utils
import models
from training_setup import TrainingSetup
import models
import torch
from tqdm import tqdm


def load_dataset():
    print('Loading dataset...')

    return utils.Mol2PropertiesDataset(
            'mlp_gnps',
            'data/pos_processed_gnps_shuffled_train.tsv',
            parser=utils.gnps_parser,
            mol_representation=utils.fingerprint,
            from_mol=0,
            to_mol=100,
            )


def load_models(hparams):
    print('Loading models...')
    _models = {}
    for hdim in hparams['hdim']:
        for n_layers in hparams['n_layers']:
            _models[f'mlp_hdim_{hdim}_layers_{n_layers}'] = models.Mol2SpecSimple(
                    molecule_dim=utils.FINGERPRINT_NBITS,
                    prop_dim=utils.SPECTRA_DIM,
                    hdim=hdim,
                    n_layers=n_layers
                    )
    return _models


def main():
    hparams = {
            'hdim': [512, 1024, 2048],
            'n_layers': [2, 3, 5],
            'batch_size': [16, 32]
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
                    optimizer=torch.optim.Adam,
                    dataloader=torch.utils.data.DataLoader
                    )

    pbar = tqdm(list(setups.items()))
    for name, tsetup in pbar:
        pbar.set_description(f'Tng: {name}')
        tsetup.train()


if __name__ == '__main__':
    main()

