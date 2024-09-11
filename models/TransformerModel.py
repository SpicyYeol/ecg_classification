import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerModel(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.embedding = nn.Linear(3, d_model)  # Embed input from 3 to d_model dimensions
        self.pos_encoder = nn.Embedding(1024, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = src.flatten(2).permute(2, 0, 1)  # (N, 3, 32, 32) -> (1024, N, 3)
        src = self.embedding(src)  # Now src is (1024, N, d_model)

        pos_indices = torch.arange(src.size(0), device=src.device).unsqueeze(1).expand(-1, src.size(1))
        pos_enc = self.pos_encoder(pos_indices)

        src = pos_enc + src  # (1024, N, d_model) + (1024, N, d_model)
        output = self.transformer_encoder(src)
        output = self.decoder(output.mean(dim=0))  # Global average pooling over the sequence length
        return output