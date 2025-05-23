import torch
import torch.nn as nn
import math
import copy

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()

        assert d_model % h == 0, "d_model must be divisible by h"

        self.h = h  # number of heads
        self.d_model = d_model
        self.d_k = d_model // h  # dimension per head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        return torch.matmul(attn, value), attn

    def forward(self, q, k, v, mask):
        # Auto-detect and flatten q, k, v if wrongly shaped (e.g., [B, H, L, D/H])
        def flatten(x, name):
            if x.dim() == 4:
                # Handle [B, H, L, D/H] or [B, L, H, D/H] — normalize to [B, L, D]
                dims = list(x.shape)
                if dims[1] == self.h:
                    # [B, H, L, D/H]
                    print(f'[WARN] Flattening {name} from [B, H, L, D/H] {x.shape}')
                    B, H, L, Dh = dims
                    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)
                elif dims[2] == self.h:
                    # [B, L, H, D/H]
                    print(f'[WARN] Flattening {name} from [B, L, H, D/H] {x.shape}')
                    B, L, H, Dh = dims
                    return x.contiguous().view(B, L, H * Dh)
            elif x.dim() > 4:
                print(f'[ERROR] Too many dimensions in {name}: {x.shape}')
                raise RuntimeError(f'Input {name} has too many dimensions: {x.shape}')
            return x

        q = flatten(q, "q")
        k = flatten(k, "k")
        v = flatten(v, "v")

        batch_size, seq_len, _ = q.size()

        # Project into heads, inferring B and L from each tensor individually
        def transform(x, linear):
            x = linear(x)                # → (B, L, D)
            B, L, D = x.size()           # grab true batch & seq length
            # reshape into (B, L, H, D/H), then (B, H, L, D/H)
            return x.view(B, L, self.h, self.d_k).transpose(1, 2)


        q = transform(q, self.q_linear)
        k = transform(k, self.k_linear)
        v = transform(v, self.v_linear)

        # 🔁 Apply scaled dot-product attention
        #print(">> SHAPES going into attention():")
        #print("input  q:", q.shape)
        #print("input  k:", k.shape)
        #print("input  v:", v.shape)
        #print("input  mask:", mask.shape if mask is not None else None)

        x, _ = self.attention(q, k, v, mask, self.dropout)

        #print(">> SHAPE from attention():", x.shape)

        # 🧩 Reshape back to [B, L, D]
        x = x.transpose(1, 2).contiguous()
        #print("x shape before reshape:", x.shape)

        if x.dim() == 5:
            if x.size(1) == self.h and x.size(2) == self.h:
                print("[FIX] Flattening x from [B, H, H, L, D/H]:", x.shape)
                B, H1, H2, L, Dh = x.shape
                # bring both head dims next to each other
                x = x.permute(0, 3, 1, 2, 4).contiguous()  # [B, L, H1, H2, Dh]
                x = x.view(B, L, H1 * H2 * Dh)              # now collapses both heads
            else:
                raise RuntimeError(f"[UNHANDLED] Unexpected 5D shape: {x.shape}")

        elif x.dim() == 4:
            B, L, H, Dh = x.shape
            x = x.contiguous().view(B, L, H * Dh)

        elif x.dim() == 3:
            print("[OK] x is already [B, L, D]:", x.shape)

        else:
            raise RuntimeError(f"[FAIL] Unexpected attention output shape: {x.shape}")



        return self.out_proj(x)






class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention_block
        self.feed_forward = feed_forward_block
        self.residuals = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residuals[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention_block
        self.cross_attention = cross_attention_block
        self.feed_forward = feed_forward_block
        self.residuals = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residuals[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residuals[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        x = self.src_embed(src)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)
        return self.decoder(x, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.encode(encoder_input, encoder_mask)
        decoder_output = self.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        return self.project(decoder_output)    


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    def clone_module(module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    attention = lambda: MultiHeadAttentionBlock(d_model, h, dropout)
    ff = lambda: FeedForwardBlock(d_model, d_ff, dropout)

    encoder_blocks = clone_module(
        EncoderBlock(d_model, attention(), ff(), dropout), N)
    decoder_blocks = clone_module(
        DecoderBlock(d_model, attention(), attention(), ff(), dropout), N)

    encoder = Encoder(d_model, encoder_blocks)
    decoder = Decoder(d_model, decoder_blocks)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    return Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)