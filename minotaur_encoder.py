import guo_et_al_unpooling as unpool
import chess_graph as cg
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


# ##### ##### ##### ##### #####
#       DGI Encoding

class NodeEncoder(nn.Module):
    def __init__(self, in_channels, gat_out_channels=8, num_gat_heads=3, activation=nn.ReLU(), mlp_hidden_dim1=(gat_out_channels * 8)):
        super(NodeEncoder, self).__init__()
        self.num_parallel_gats = 8
        self.activation = activation

        # ----- First Parallel GAT Layer -----
        self.gat_layers1 = nn.ModuleList([
            GATConv(in_channels, gat_out_channels, heads=num_gat_heads, concat=False, dropout=0.1)
            for _ in range(self.num_parallel_gats)
        ])
        gat_output_dim1 = gat_out_channels

        # MLP to transform concatenated outputs of the first 8 GATs
        concat_dim1 = gat_output_dim1 * self.num_parallel_gats  # 64
        mlp_out_dim1 = int(concat_dim1 / 2)  # 32
        self.mlp1 = nn.Sequential(
            nn.Linear(concat_dim1, mlp_hidden_dim1),
            activation,
            nn.Linear(mlp_hidden_dim1, mlp_out_dim1)
        )

        # ----- Second Parallel GAT Layer -----
        gat_output_dim2 = (mlp_out_dim1 * 2) # 64
        self.gat_layers2 = nn.ModuleList([
            GATConv(mlp_out_dim1, gat_output_dim2, heads=num_gat_heads, concat=False, dropout=0.1)
            for _ in range(self.num_parallel_gats)
        ])

        # MLP to transform concatenated outputs of the second set of GATs
        concat_dim2 = gat_output_dim2 * self.num_parallel_gats  # 512
        mlp_hidden_dim2 = concat_dim2 * 8  # 4096
        mlp_out_dim2 = int(concat_dim2 / 4)  # 128
        self.mlp2 = nn.Sequential(
            nn.Linear(concat_dim2, mlp_hidden_dim2),
            activation,
            nn.Linear(mlp_hidden_dim2, mlp_out_dim2)
        )

        # ----- Third Parallel GAT Layer -----
        gat_output_dim3 = (mlp_out_dim2 * 2)  # 256
        self.gat_layers3 = nn.ModuleList([
            GATConv(mlp_out_dim2, gat_output_dim3, heads=num_gat_heads, concat=False, dropout=0.1)
            for _ in range(self.num_parallel_gats)
        ])

        # MLP to transform concatenated outputs of the third set of GATs
        concat_dim3 = gat_output_dim3 * self.num_parallel_gats  # 2064
        mlp_hidden_dim3 = concat_dim3 * 8  # 16,512
        mlp_out_dim3 = int(concat_dim3 / 4)  # 512
        self.mlp3 = nn.Sequential(
            nn.Linear(concat_dim3, mlp_hidden_dim3),
            activation,
            nn.Linear(mlp_hidden_dim3, mlp_out_dim3)
        )

    def forward(self, x, edge_index_list):
        # --- First hop ---
        h = [self.activation(gat(x, edge_index_list[i])) for i, gat in enumerate(self.gat_layers1)]
        h = self.activation(self.mlp1(torch.cat(h, dim=-1)))

        # --- Second hop ---
        h = [self.activation(gat(h, edge_index_list[i])) for i, gat in enumerate(self.gat_layers2)]
        h = self.activation(self.mlp2(torch.cat(h, dim=-1)))

        # --- Third hop ---
        h = [self.activation(gat(h, edge_index_list[i])) for i, gat in enumerate(self.gat_layers3)]
        h = self.mlp3(torch.cat(h, dim=-1))  # <- final activation already inside mlp3

        return h  # shape: [64, 512]


class GlobalSummarizer(nn.Module):
    def __init__(self):
        super(GlobalSummarizer, self).__init__()

        self.gat_layers1 = nn.ModuleList([
            GATConv(
                in_channels=8,  # your per-square feature length
                out_channels=8,  # 8
                heads=8,  # 8 heads
                concat=True,  # ✱ important for 64-d output
                dropout=0.1
            ) for _ in range(8)  # one per edge-type
        ])



    def forward(self):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        pass




# ----- ----- -----
# Program Body

if __name__ == "__main__":
    # Initialize the node encoder

    # Initialize the global summarizer (latent space encoder)

    # Initialize the discriminator

    # Get the list of positions

    # For each position
        # Get the global summary vector for the current true position

        # Choose randomly 50/50 whether to perturb the position
        # Keep track in a boolean

        # Pick a random square
        # Get the node encoding of the random square for the position (maybe perturbed)

        # Feed the global summary of the true position and the node encoding to the discriminator
        # Check its output vs the boolean that says whether I perturbed or not





