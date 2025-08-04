import torch
import torch.nn as nn
from torch_geometric.data import Data


# Helper function for splitting children
def _split_feats(self, x):
    # x: [N, d]
    d = x.size(-1)
    ds, D = d // 2, d // 4
    # PS1 = x[:ds+D]      (keep first 3/4)
    # PS2 = x[:ds] + x[ds+D:]  (skip middle quarter)
    part1 = x[:, :ds + D]
    part2 = torch.cat([x[:, :ds], x[:, ds + D:]], dim=-1)
    return part1, part2


class GuoEtAlUnpool(nn.Module):
    def __init__(self, dx=64, dw=8, dy=64):
        super(GuoEtAlUnpool, self).__init__()
        # MLP-R: Outputs probability of unpooling a node
        self.decide_split   = MLP(dx, (dx * 2), 1)                # Input dim, hidden dim, 1 logit [0,1]
        # MLP-y: Child node feature generator
        self.node_mlp       = MLP(int(0.75 * dx), (dx * 2), dy)   # Project 3/4, hidden dim, output dim
        # MLP-IA: Decides whether to intra-link children
        self.intra_mlp      = MLP(dx, (dx * 2), 1)                # Input dim, hidden dim, 1 logit [0,1]
        # MLP-IE: Decides how to rewire an old edge
        self.inter_mlp      = MLP(2 * dy + dw + dx, (dx * 2), 3)  # Input dim (children, edge, neighbors), hidden dim, 3 logits [0,1]
        # MLP-C: Connectivity enforcer
        self.conn_mlp       = MLP(2 * dy + dw + dx, (dx * 2), 1)  # Input dim (children, edge, neighbors), hidden dim, 1 logit [0,1]
        # MLP-u: Generate new edge features
        self.edge_feat_mlp  = MLP(dy, (dx * 2), (dw * 2))         # Input dim (sum of 2 vectors), hidden dim, output edge feature dim

    def forward(self, data, sample=True):
        logP = data.x.new_zeros(1)

        # 1. split decisions
        split_logits = self.decide_split(data.x).squeeze(-1)
        split_probs = split_logits.sigmoid()
        if sample:
            split_mask = torch.bernoulli(split_probs)
            logP += (split_mask * split_probs.log() +
                     (1 - split_mask) * (1 - split_probs).log()).sum()
        else:
            split_mask = (split_probs > 0.5).float()

        # 2. feature slicing / child creation ...
        # (use _split_feats helper)

        # 3. intra-links, inter-links, connectivity
        # (vectorised, accumulate logP)

        new_data = Data()

        return new_data, logP



