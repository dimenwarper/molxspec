import utils
import models
import graphs
from training_setup import TrainingSetup
import torch_geometric
import models
from torch import optim
from tqdm import tqdm


def load_dataset():
    print('Loading dataset...')

    return utils.Mol2PropertiesDataset(
            'egnn_gnps',
            (
                'data/pos_processed_gnps_shuffled_with_3d_train.tsv',
                'data/pos_processed_gnps_shuffled_with_3d_train.sdf',
            ),
            parser=utils.gnps_parser_3d,
            mol_representation=graphs.mol_to_torch_geom,
            #from_mol=0,
            #to_mol=100,
            use_cache=True,
            mol_rep_kwargs={'add_positions': True}
            )


def load_models(hparams):
    print('Loading models...')
    _models = {}
    for hdim in hparams['hdim']:
        for n_layers  in hparams['n_layers']:
            _models[f'egnn_hdim_{hdim}_layers_{n_layers}'] = models.Mol2SpecEGNN(
                    molecule_dim=graphs.NUM_NODE_FEATURES,
                    prop_dim=utils.SPECTRA_DIM,
                    hdim=hdim,
                    edge_dim=graphs.NUM_EDGE_FEATURES,
                    n_layers=n_layers
                    )
    return _models


def main():
    hparams = {
            'hdim': [256, 512, 1024],
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
                    dataloader=torch_geometric.loader.DataLoader
                    )

    pbar = tqdm(list(setups.items()))
    for name, tsetup in pbar:
        pbar.set_description(f'Tng: {name}')
        tsetup.train()


if __name__ == '__main__':
    main()
