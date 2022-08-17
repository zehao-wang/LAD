from torch import nn 
import torch.nn.functional as F 
from .vilmodel import BertXAttention, BertLayerNorm, ClsPrediction
import torch

class NodeDistReg(nn.Module):
    def __init__(self, input_size, config, hidden_size=None):
        super().__init__()
        self.config = config
        if hidden_size is None:
            hidden_size = input_size 
        
        self.gd_dropout = None
        if config.gd_feat_dropout:
            self.gd_dropout = nn.Dropout(p=config.gd_feat_dropout)

        self.cross_attn = BertXAttention(config, ctx_dim=768)
        self.score_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            BertLayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, 1)
        )
        self.dreamer_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
        self.dist_sap_head = ClsPrediction(self.config.hidden_size)
            
    def forward(self, x, ins_img_features):
        ins_img_features = torch.from_numpy(ins_img_features).float().to(x.device)
        if self.gd_dropout is not None:
            ins_img_features=self.gd_dropout(ins_img_features)

        attention_output, attention_scores = self.cross_attn(x, ins_img_features)
        fuse_weight = torch.sigmoid(self.dreamer_fuse_linear(
            torch.cat([x[:, 0], attention_output[:, 0]], 1)
        ))
        return None, self.dist_sap_head(attention_output).squeeze(2), fuse_weight
