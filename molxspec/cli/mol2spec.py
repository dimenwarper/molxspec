from argparse import ArgumentParser
from molxspec import mol2spec, utils


def clargs():
    parser = ArgumentParser(description='Predict positive mode MS2 spectra from a file of SMILES-encoded small molecules')
    parser.add_argument('smilesfile', help='File with SMILES encoded small molecules, one per line')
    parser.add_argument('outfile', help='File to write spectra to')
    parser.add_argument(
        '--model', 
        default='mlp', 
        help='Model to use for prediction, must be one of mlp, bert, egnn, or gcn. Default is mlp'
        )
    parser.add_argument(
        '--fraglevel', 
        type=int, 
        default=4, 
        help='Fragmentation level, 0-4, higher is more fragmented. Default is 4'
        )
    parser.add_argument(
        '--adduct', 
        default='M+H', 
        help=f'Adduct type to predict, must be one of {",".join(utils.ADDUCTS)}. Default is M+H'
        )

    args = parser.parse_args()
    if args.fraglevel > 4 or args.fraglevel < 0:
        print('Invalid fragementation level, should be 0-4')
    if args.adduct not in utils.ADDUCTS:
        print(f'Invalid adduct, should be one of {utils.ADDUCTS}')
    return args


def main():
    args = clargs()
    with open(args.smilesfile) as fl:
        smiless = [smiles.strip() for smiles in fl.readlines()]
    specs = mol2spec.predict(smiless, -args.fraglevel, args.adduct, mol2spec.ModelType(args.model))
    with open(args.outfile, 'w') as ofl:
        for smiles, spec in zip(smiless, specs):
            ofl.write(f'{smiles}\t{spec.tolist()}\n')
    

if __name__ == '__main__':
    main()