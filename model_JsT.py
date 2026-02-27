import torch
import torch.nn as nn
import math

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
DROPOUT_RATE = 0.5 

class DiffusionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_len, pad_idx, dropout_rate=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Project continuous noisy one-hot (B, L, vocab) -> (B, L, embedding_dim)
        # Must be defined in __init__ so state_dict keys match trained checkpoints.
        self.input_projection = nn.Linear(vocab_size, embedding_dim)
        
        # 2. Positional Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embedding_dim))
        
        # 3. Time Embedding (Sinusoidal)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8, # 8 heads
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6) # 6 layers
        
        # 5. Output Layer (Modified for x0-prediction)
        # 创新点：为了更精准地预测 clean data (x0)，我们强化了输出层
        # 移除了这里的 Dropout，并加了 LayerNorm 稳定 Logits
        self.output_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.GELU(),
            # 注意：这里不再使用 Dropout，因为我们要直接回归原始信号
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_timestep_embedding(self, timesteps, embedding_dim):
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, x, t):
        # x shape: [batch, seq_len, vocab_size] (Continuous/Noisy One-hot)
        # t shape: [batch]
        
        x_emb = self.input_projection(x)
        
        # Add Time Embedding
        t_emb = self.get_timestep_embedding(t, x_emb.shape[-1])
        t_emb = self.time_mlp(t_emb)
        
        # Combine
        x_emb = x_emb + self.pos_embedding[:, :x.shape[1], :] + t_emb.unsqueeze(1)
        x_emb = self.dropout(x_emb)
        
        # Transformer
        # mask logic needs to be handled if padding exists, omitted for brevity as per original context
        h = self.transformer(x_emb)
        
        # Output
        # 输出的是预测的 x_start (logits)，而不是噪声
        prediction = self.output_head(h)
        return prediction

def get_diffusion_beta_schedule(num_diffusion_steps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_diffusion_steps)