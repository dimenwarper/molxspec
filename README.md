# molxspec
Machine learning models to convert molecules to ESI mass spectra (and maybe back again in a future version) trained on [GNPS data](https://gnps.ucsd.edu/). Currently the following models are available:

## mol2spec

| model | description                                                 | hidden unit dim | num layers | 
|-------|-------------------------------------------------------------|-----------------|------------|
| mlp   | MLP with residual blocks trained on 1024 Morgan fingerprints| 1024            | 6          |
| gcn   | Simple GCN with deepchem like node features                 | 1024            | 3          |
| egnn  | Equivariant GNN trained on (RDkit optimized) 3D structures  | 1024            | 2          |
| bert  | MLP trained on representations from the (smaller) ChemBERTa SMILES model | 1024 | 6        |

## Installation

You need to have `torch` and `torch_geometric` installed. I don't provide these as part of the dependencies since `torch_geometric` installs depends a lot on your CUDA and `torch` setup. To install `torch_geometric` from scratch [use their documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html); e.g. can do it with pip using their wheels:

```
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cpu.html
```

Once `torch` and `torch_geometric` are installed, you can install `molxspec`:

```
pip install https://github.com/dimenwarper/molxspec/releases/download/v.0.1.0/molxspec-0.1.0-py3-none-any.whl
```

## Usage

You can predict spectra from the command line:

```
mol2spec --model [mlp | gcn | egnn | bert] input_smiles.txt output.txt
```

Where `input_smiles.txt` is a file containing one molecule SMILES for each line. For the `egnn` model, molecules will have their 3D structure computed and optimized automatically using RDKit. First time use will download the pretrained models automatically, which can take some time, though it is a one-time thing only.

You can also predict spectra programmatically:

```python
from molxspec import mol2spec
dict_of_smiles_and_spectra = mol2spec.predict(list_of_smiles)
```
