import torch
from torch.nn import Linear

from torch_geometric.nn import GraphConv, dense_mincut_pool
from torch_geometric import utils
from torch_geometric.nn import Sequential

torch.manual_seed(0)


class MinCutNet(torch.nn.Module):
    def __init__(
        self,
        mp_units,
        mp_act,
        in_channels,
        n_clusters,
        mlp_units=[],
        mlp_act="Identity",
    ):
        super().__init__()

        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)

        mp = [
            (GraphConv(in_channels, mp_units[0]), "x, edge_index, edge_weight -> x"),
            mp_act,
        ]
        for i in range(len(mp_units) - 1):
            mp.append(
                (
                    GraphConv(mp_units[i], mp_units[i + 1]),
                    "x, edge_index, edge_weight -> x",
                )
            )
            mp.append(mp_act)
        self.mp = Sequential("x, edge_index, edge_weight", mp)
        out_chan = mp_units[-1]

        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))

    def forward(self, x, edge_index, edge_weight):
        x = self.mp(x, edge_index, edge_weight)
        s = self.mlp(x)
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        x_pool, adj_pool, mc_loss, o_loss = dense_mincut_pool(x, adj, s)

        return torch.softmax(s, dim=-1), mc_loss, o_loss, x_pool, adj_pool
