import torch
from torch import nn

class MultimodalRegressor(nn.Module):
    def __init__(self, text_dim, tab_dim, hidden_dim=256):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # oncatenate
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, text, tab):
        text_emb = self.text_encoder(text)
        tab_emb = self.tab_encoder(tab)
        combined = torch.cat([text_emb, tab_emb], dim=1)
        return self.fusion(combined)