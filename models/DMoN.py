import torch
from torch.nn import Linear
from torch_geometric import utils
from torch_geometric.nn import Sequential, DMoNPooling, GCNConv

torch.manual_seed(0)


class DMoNNet(torch.nn.Module):
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
            (
                GCNConv(in_channels, mp_units[0], normalize=False, cached=False),
                "x, edge_index, edge_weight -> x",
            ),
            mp_act,
        ]
        for i in range(len(mp_units) - 1):
            mp.append(
                (
                    GCNConv(
                        mp_units[i], mp_units[i + 1], normalize=False, cached=False
                    ),
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

        self.dmon_pooling = DMoNPooling(channels=mp_units[-1], k=n_clusters)

    def forward(self, x, edge_index, edge_weight):
        x = self.mp(x, edge_index, edge_weight)
        s = self.mlp(x)
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, _, spectral_loss, ortho_loss, cluster_loss = DMoNPooling(x, adj)

        return torch.softmax(s, dim=-1), spectral_loss, ortho_loss, cluster_loss
