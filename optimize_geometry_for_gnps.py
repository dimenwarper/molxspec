import subprocess
from tqdm import tqdm

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

def main():
    with open('data/pos_processed_gnps_shuffled.txt') as fl, \
        open('data/pos_processed_gnps_shuffled.sdf', 'w') as sdffl, \
        open('data/pos_processed_gnps_shuffled.failed_sdf', 'w') as failedfl, \
        open('data/pos_processed_gnps_shuffled_with_3d.sdf', 'w') as outfl:
        for i, line in tqdm(enumerate(fl), total=160000):
            smiles = line.strip().split('\t')[1]
            output = gen3d(smiles)
            if output is None:
                failedfl.write(f'{i}\n')
            else:
                sdffl.write(output + '\n')
                outfl.write(line)

if __name__ == '__main__':
    main()