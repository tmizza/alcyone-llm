import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size=25_000, d_model=512, n_heads=8, n_layers=12, max_len=256):
        super().__init__()

        # ✅ Token & positional embeddings (Same as Before)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        # ✅ Transformer Layers (Same as Before, Just Adjusted for 100M Params)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True
            ) for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        x = self.token_embedding(x) + self.position_embedding(pos)

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.layer_norm(x)
        return self.fc_out(x)

# ✅ Initialize the model (Same as Before, Just 100M Params)
model = MiniGPT()
print(f"✅ MiniGPT Model Initialized! Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
