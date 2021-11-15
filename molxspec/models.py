import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import egnn
import utils
from transformers import AutoModelWithLMHead
from itertools import chain


class ResBlock(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class Mol2SpecSimple(nn.Module):
    def __init__(self, molecule_dim: int, prop_dim: int, hdim: int, n_layers: int):
        super().__init__()
        self.kwargs = dict(
            molecule_dim=molecule_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers
        )

        self.meta = nn.Parameter(torch.empty(0))
        self.molecule_dim = molecule_dim
        self.prop_dim = prop_dim
        dropout_p = 0.1
        res_blocks = [
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(molecule_dim, hdim),
                    nn.SiLU(),
                    nn.Linear(hdim, molecule_dim),
                )
            )
            for _ in range(n_layers)
        ]
        self.mlp_layers = nn.Sequential(
            *res_blocks,
            nn.Linear(molecule_dim, prop_dim),
        )


    def forward(self, mol_vec):
        mol_vec = mol_vec.type(torch.FloatTensor).to(self.meta.device)
        mz_res = self.mlp_layers(mol_vec)
        return mz_res



class Mol2SpecGraph(nn.Module):
    def __init__(self, node_feature_dim: int, graph_feature_dim: int, prop_dim: int, hdim: int, n_layers: int):
        super().__init__()
        self.kwargs = dict(
            node_feature_dim=node_feature_dim,
            graph_feature_dim=graph_feature_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers
        )
        self.meta = nn.Parameter(torch.empty(0))

        gcn_middle_layers = []
        for _ in range(n_layers):
            gcn_middle_layers += [
                (gnn.GCNConv(hdim, hdim), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
            ]

        self.gcn_layers = gnn.Sequential(
            'x, edge_index, batch',
            [
                (gnn.GCNConv(node_feature_dim, hdim), 'x, edge_index -> x'),
                nn.ReLU(inplace=True),
            ] + gcn_middle_layers + [
                (gnn.global_max_pool, 'x, batch -> x'),
            ]
        )

        # Would have liked training more than 1 layer, but my current setup is sooo slow
        dropout_p = 0.1
        res_blocks = [
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(hdim + graph_feature_dim, hdim),
                    nn.SiLU(),
                    nn.Linear(hdim, hdim + graph_feature_dim),
                )
            )
            for _ in range(1)
        ]

        self.head_layers = nn.Sequential(
            *res_blocks,
            nn.Linear(hdim + graph_feature_dim, prop_dim),
        )


    def forward(self, gdata):
        gdata = gdata.to(self.meta.device)
        x = self.gcn_layers(gdata.x, gdata.edge_index, gdata.batch)

        batch_size = x.shape[0]
        frag_levels = gdata.frag_levels.reshape([batch_size, gdata.frag_levels.shape[0] // batch_size])
        adduct_feats = gdata.adduct_feats.reshape([batch_size, gdata.adduct_feats.shape[0] // batch_size])

        x = torch.cat((x, frag_levels, adduct_feats), axis=1)
        x = self.head_layers(x)
        return x


class Mol2SpecEGNN(nn.Module):
    def __init__(self, node_feature_dim: int, graph_feature_dim: int, prop_dim: int, hdim: int, edge_dim: int, n_layers: int):
        super().__init__()
        self.kwargs = dict(
            node_feature_dim=node_feature_dim,
            graph_feature_dim=graph_feature_dim,
            prop_dim=prop_dim,
            hdim=hdim,
            edge_dim=edge_dim,
            n_layers=n_layers,
        )
        self.meta = nn.Parameter(torch.empty(0))

        self.egnn = egnn.EGNN(
            in_node_nf=node_feature_dim, 
            hidden_nf=hdim,
            out_node_nf=hdim, 
            in_edge_nf=edge_dim,
            n_layers=n_layers
            )

        self.pool_layers = gnn.Sequential('x, batch', [
            (gnn.global_max_pool, 'x, batch -> x'),
            ])

        # Would have liked training more than 1 layer, but my current setup is sooo slow
        dropout_p = 0.1
        res_blocks = [
            ResBlock(
                nn.Sequential(
                    nn.Dropout(p=dropout_p),
                    nn.Linear(hdim + graph_feature_dim, hdim),
                    nn.SiLU(),
                    nn.Linear(hdim, hdim + graph_feature_dim),
                )
            )
            for _ in range(1)
        ]

        self.head_layers = nn.Sequential(
            *res_blocks,
            nn.Linear(hdim + graph_feature_dim, prop_dim),
        )

    def forward(self, gdata):
        gdata = gdata.to(self.meta.device)
        x, _ = self.egnn(gdata.x, gdata.pos, gdata.edge_index, gdata.edge_attr)
        x = self.pool_layers(x, gdata.batch)

        batch_size = x.shape[0]
        frag_levels = gdata.frag_levels.reshape([batch_size, gdata.frag_levels.shape[0] // batch_size])
        adduct_feats = gdata.adduct_feats.reshape([batch_size, gdata.adduct_feats.shape[0] // batch_size])

        x = torch.cat((x, frag_levels, adduct_feats), axis=1)
        x = self.head_layers(x)
        return x


class Mol2SpecBERT(nn.Module):
    def __init__(self, prop_dim: int, hdim: int, n_layers: int):
        super().__init__()
        self.kwargs = dict(
            prop_dim=prop_dim,
            hdim=hdim,
            n_layers=n_layers,
        )
        self.meta = nn.Parameter(torch.empty(0))
        self.pretrained = AutoModelWithLMHead.from_pretrained(utils.CHEMBERTA_MODEL)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.in_dim = int(self.pretrained.lm_head.dense.in_features)
        self.head = nn.Sequential(
            nn.Linear(self.in_dim, hdim),
            nn.SiLU(),
            *chain(*[(nn.Linear(hdim, hdim), nn.SiLU()) for _ in range(n_layers)]), 
            nn.Linear(hdim, prop_dim)
        )
    
    def forward(self, x):
        x = {k: v.to(self.meta.device).squeeze(dim=1) for k, v in x.items()}
        x = self.pretrained(**x, output_hidden_states=True).hidden_states[-1].mean(axis=1)
        x = self.head(x)
        return x