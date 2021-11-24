# molxspec
Machine learning models to convert molecules to ESI mass spectra (and maybe back again in a future version) trained on [GNPS data](https://gnps.ucsd.edu/). Currently the following models are available:

## mol2spec

| model | description                                                 | hidden unit dim | num layers | 
|-------|-------------------------------------------------------------|-----------------|------------|
| mlp   | MLP with residual blocks trained on 1024 Morgan fingerprints| 1024            | 6          |
| gcn   | Simple GCN with deepchem like node features                 | 1024            | 3          |
| egnn  | Equivariant GNN trained on (RDkit optimized) 3D structures  | 1024            | 2          |
| bert  | MLP trained on representations from the (smaller) ChemBERTa SMILES model | 1024 | 6        |

## Installation and usage

If you just want to predict spectra, then the easiest way is to just pip install (we don't currently have this indexed in pypi due to non-pypi torch dependencies):

```
pip install https://github.com/dimenwarper/molxspec/releases/download/v.0.1.0/molxspec-0.1.0-py3-none-any.whl
```

And then do it via command line:

```
mol2spec --model [mlp | gcn | egnn | bert] input_smiles.txt output.txt
```

Where `input_smiles.txt` is a file containing one molecule SMILES for each line
First time use will download the pretrained models automatically

You can also do it programmatically after you install:

```python
from molxspec import mol2spec
dict_of_smiles_and_spectra = mol2spec.predict(list_of_smiles)
```

## Installation fails

Installation might fail if you are not on an x86 linux platform (e.g. colab):
```
ERROR: torch_sparse...is not a supported wheel on this platform.
```

in which case you might just want to clone the repository and create a conda environment with the provided `environment.yaml`:

```
git clone https://github.com/dimenwarper/molxspec.git
cd molxspec
conda env create --name molxspec --file=environments.yml
conda activate molxspec
```

