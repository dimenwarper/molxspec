import utils
import models
from training_setup import TrainingSetup



def load_datasets():
    print('Loading dataset...')

    return utils.Mol2PropertiesDataset(
            'data/pos_processed_gnps_shuffled.txt',
            parser=utils.gnps_parser,
            mol_representation=utils.fingerprint,
            #from_mol=0,
            #to_mol=65000
            )


def load_models(hparams):
    print('Loading models...')
    _models = {}
    for hdim in hparams['hdim']:
        _models[f'simple_{hdim}'] = models.Mol2SpecSimple(
                molecule_dim=utils.FINGERPRINT_NBITS,
                prop_dim=utils.SPECTRA_DIM, hdim=hdim
                )
    return _models

    models = {}
    return models


def main():
    dataset = load_dataset()
    HDIMS = [1000, 2000]
    hparams = {
            'hdim': [1000, 2000],
            'batch_size': [16, 23]
            }
    _models = load_models(hparams)


    setups = {}
    for bsz in hparams['batch_size']:
        for mname, model in _models.items():
            setup_name = f'model_{mname}_bs_{bsz}_w_dropout_adam'
            setups[setup_name] = TrainingSetup(
                    model=model,
                    dataset=DATASETS['fingerprint'],
                    outdir=f'runs/{setup_name}',
                    batch_size=bsz,
                    n_epochs=100,
                    optimizer=optim.Adam
                    )

    pbar = tqdm(list(setups.items()))
    for name, tsetup in pbar:
        pbar.set_description(f'Tng: {name}')
        tsetup.train()


if __name__ == '__main__':
    main()
