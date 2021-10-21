import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class ResBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class Mol2SpecSimple(nn.Module):
    def __init__(self, molecule_dim: int, prop_dim: int, hdim: int = 1000):
        super().__init__()
        self.meta = nn.Parameter(torch.empty(0))
        self.molecule_dim = molecule_dim
        self.prop_dim = prop_dim
        nl = nn.SiLU()
        dropout_p = 0.1
        self.mlp_layers = nn.Sequential(
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(molecule_dim, hdim),
                    nl,
                    nn.Linear(hdim, molecule_dim),
                )
            ),
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(molecule_dim, hdim),
                    nl,
                    nn.Linear(hdim, molecule_dim),
                )
            ),
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(molecule_dim, hdim),
                    nl,
                    nn.Linear(hdim, molecule_dim),
                )
            ),
            nn.Linear(molecule_dim, prop_dim),
        )


    def forward(self, mol_vec):
        mol_vec = mol_vec.type(torch.FloatTensor).to(self.meta.device)
        mz_res = self.mlp_layers(mol_vec)
        return mz_res



class Mol2SpecGraph(nn.Module):
    def __init__(self, molecule_dim: int, prop_dim: int, hdim: int = 32):
        super().__init__()
        self.meta = nn.Parameter(torch.empty(0))

        self.layers = gnn.Sequential('x, edge_index, batch', [
            (gnn.GCNConv(molecule_dim, hdim), 'x, edge_index -> x1'),
            nn.ReLU(inplace=True),
            (gnn.GCNConv(hdim, hdim), 'x1, edge_index -> x2'),
            nn.ReLU(inplace=True),
            (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            (gnn.JumpingKnowledge('cat', hdim, num_layers=2), 'xs -> x'),
            (gnn.global_add_pool, 'x, batch -> x'),
            nn.Linear(hdim * 2, hdim // 2),
            nn.Dropout(p=0.2),
            nn.Linear(hdim // 2, prop_dim),
            ])


    def forward(self, gdata):
        gdata = gdata.to(self.meta.device)
        x = self.layers(gdata.x, gdata.edge_index, gdata.batch)
        return x
