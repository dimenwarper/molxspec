from molxspec import utils, models, chemberta
from molxspec.training_setup import TrainingSetup, cli
import torch
from tqdm import tqdm


def load_dataset():
    print('Loading dataset...')

    return utils.Mol2SpecDataset(
            'bert_gnps',
            'data/pos_processed_gnps_shuffled_with_3d_train.tsv',
            parser=utils.gnps_parser,
            mol_representation=chemberta.chemberta_representation,
            #from_mol=0,
            #to_mol=50000,
            use_cache=True
            )


def load_models(hparams):
    print('Loading models...')
    _models = {}
    for hdim in hparams['hdim']:
        for n_layers in hparams['n_layers']:
            _models[f'bert_hdim_{hdim}_layers_{n_layers}'] = models.Mol2SpecSimple(
                    molecule_dim=chemberta.get_model_dim() + len(utils.FRAGMENT_LEVELS) + len(utils.ADDUCTS),
                    prop_dim=utils.SPECTRA_DIM,
                    hdim=hdim,
                    n_layers=n_layers
                    )
    return _models

SCAN_HPARAMS = {
    'hdim': [512, 1024, 2048],
    'n_layers': [1, 2, 3, 5, 6, 7],
    'batch_size': [16, 32, 64, 128, 256]
}

PROD_HPARAMS = {
    'hdim': [1024],
    'n_layers': [6],
    'batch_size': [256]
}

def main():
    setup_args, clargs, hparams = cli(SCAN_HPARAMS, PROD_HPARAMS)
    dataset = load_dataset()
    _models = load_models(hparams)
    setup_args['n_epochs'] = min(100, setup_args['n_epochs'])


    setups = {}
    for bsz in hparams['batch_size']:
        for mname, model in _models.items():
            suffix = '_prod' if clargs.prod else ''
            setup_name = f'model_{mname}_bs_{bsz}_adam{suffix}'
            setups[setup_name] = TrainingSetup(
                    model=model,
                    dataset=dataset,
                    outdir=f'runs/{setup_name}',
                    batch_size=bsz,
                    dataloader=torch.utils.data.DataLoader,
                    **setup_args
                    )

    pbar = tqdm(list(setups.items()))
    for name, tsetup in pbar:
        pbar.set_description(f'Tng: {name}')
        tsetup.train()


if __name__ == '__main__':
    main()

