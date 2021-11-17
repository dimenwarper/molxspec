import subprocess
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


def rdkit3d(smiles):
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(m, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(m)
        return Chem.MolToMolBlock(m) + '\n$$$$$\n'
    except ValueError:
        return None


def gen3d(smiles):
    stdcmd = f'timeout 60 obabel -:"{smiles}" -osdf --gen3d'
    fastestcmd = f'timeout 60 obabel -:"{smiles}" -osdf --gen3d --fastest'

    for cmd in [stdcmd, fastestcmd]:
        try:
            toret = subprocess.check_output(cmd, shell=True).decode('utf-8')
            if len(toret.strip()) == 0:
                raise ValueError
            else:
                return toret
        except (subprocess.CalledProcessError, ValueError):
            pass
    return None


def optimize(infname, sdffname, outfname, flen):
    with open(infname) as fl, \
        open(sdffname, 'w') as sdffl, \
        open(f'{outfname}.failed', 'w') as failedfl, \
        open(outfname, 'w') as outfl:
        for i, line in tqdm(enumerate(fl), total=flen):
            smiles = line.strip().split('\t')[-1]
            output = rdkit3d(smiles)
            if output is None:
                failedfl.write(f'{i}\n')
            else:
                sdffl.write(output)
                outfl.write(line)


def main():
    optimize(
        '../data/pos_processed_gnps_shuffled_train.tsv',
        '../data/pos_processed_gnps_shuffled_with_3d_train.sdf',
        '../data/pos_processed_gnps_shuffled_with_3d_train.tsv',
        140000,
    )

    optimize(
        '../data/pos_processed_gnps_shuffled_validation.tsv',
        '../data/pos_processed_gnps_shuffled_with_3d_validation.sdf',
        '../data/pos_processed_gnps_shuffled_with_3d_validation.tsv',
        5000
    )


if __name__ == '__main__':
    main()