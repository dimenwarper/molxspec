from molxspec import utils, models, graphs
from molxspec.training_setup import TrainingSetup, cli
import torch_geometric
from tqdm import tqdm


def load_dataset():
    print('Loading dataset...')

    return utils.Mol2SpecDataset(
            'graph_gnps',
            'data/pos_processed_gnps_shuffled_with_3d_train.tsv',
            parser=utils.gnps_parser,
            mol_representation=graphs.mol_to_torch_geom,
            #from_mol=0,
            #to_mol=100,
            #use_cache=True,
            )


def load_models(hparams):
    print('Loading models...')
    _models = {}
    for hdim in hparams['hdim']:
        for n_layers in hparams['n_layers']:
            _models[f'graphconv_hdim_{hdim}_layers_{n_layers}'] = models.Mol2SpecGraph(
                    node_feature_dim=graphs.NUM_NODE_FEATURES,
                    graph_feature_dim=len(utils.FRAGMENT_LEVELS) + len(utils.ADDUCTS),
                    prop_dim=utils.SPECTRA_DIM,
                    hdim=hdim,
                    n_layers=n_layers
                    )
    return _models

SCAN_HPARAMS = {
            'hdim': [1024, 2048],
            'batch_size': [32, 64, 128],
            'n_layers': [1, 2, 3, 4]
}

PROD_HPARAMS = {
    'hdim': [1024],
    'n_layers': [3],
    'batch_size': [32]
}

def main():
    setup_args, clargs, hparams = cli(SCAN_HPARAMS, PROD_HPARAMS)
    dataset = load_dataset()
    _models = load_models(hparams)

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
                    dataloader=torch_geometric.loader.DataLoader,
                    lr=3e-4,
                    **setup_args
                    )

    pbar = tqdm(list(setups.items()))
    for name, tsetup in pbar:
        pbar.set_description(f'Tng: {name}')
        tsetup.train()


if __name__ == '__main__':
    main()
