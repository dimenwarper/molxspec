import ijson
from rdkit import Chem
from tqdm import tqdm
from rdkit import RDLogger
import numpy as np
import json
from molxspec import utils


RDLogger.DisableLog('rdApp.*')  
import warnings

def main():
    objects = ijson.items(open('data/ALL_GNPS.json'), 'item')
    with open('data/pos_processed_gnps.txt', 'w') as pos_fl, open('data/neg_processed_gnps.txt', 'w') as neg_fl, open('data/invalid_entries', 'w') as prob_fl:
        for o in tqdm(objects, total=650000):
            if 'esi' in o['Ion_Source'].lower() and o['ms_level'] == '2' and o['Smiles'].strip() != 'N/A' and len(o['Smiles'].strip()) > 0 and o['Adduct'] in utils.ADDUCTS:     
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mol = Chem.MolFromSmiles(o['Smiles'])
                if mol is None:
                    prob_fl.write(f'{o["spectrum_id"]}\n')
                else:
                    spec = np.array(json.loads(o['peaks_json']))
                    if len(spec) > 0 and len(spec.shape) == 2 and spec.shape[0] >= 10 and  (spec <= 0).sum() == 0 and (spec[:, 0] > utils.MAX_MZ).sum() == 0:
                        if o['Ion_Mode'] == 'Positive':
                            pos_fl.write(f"{o['spectrum_id']}\t{o['Adduct']}\t{o['peaks_json']}\t{o['Smiles']}\n")
                        elif o['Ion_Mode'] == 'Negative':
                            neg_fl.write(f"{o['spectrum_id']}\t{o['Adduct']}\t{o['peaks_json']}\t{o['Smiles']}\n")
                    

if __name__ == '__main__':
    main()