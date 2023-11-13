import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model,
            max_len
        ):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).expand(-1, d_model)
        exponent = torch.arange(0, d_model).unsqueeze(0)
        exponent = 10_000 ** (exponent / d_model)
        pe = position / exponent
        pe[:, ::2] = torch.sin(pe[:, ::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + x + self.pe[:, :x.size(1)]


class Attention(nn.Module):
    def __init__(
            self,
            heads,
            d_model
        ):
        super(Attention, self).__init__()
        self.heads = heads 
        self.d_model = d_model

        self.d_k = d_model // heads # for keys and queries
        self.d_v = d_model // heads # for values

        self.keys_transformations = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(self.heads)])
        self.queries_transformations = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(self.heads)])
        self.values_transformations = nn.ModuleList([nn.Linear(d_model, self.d_v, bias=False) for _ in range(self.heads)])

        self.fc_out = nn.Linear(d_model, d_model)


    def forward(self, queries, keys, values, mask=None):
        # 1. Transform input data
        keys = [transformation(keys) for transformation in self.keys_transformations]
        queries = [transformation(queries) for transformation in self.queries_transformations]
        values = [transformation(values) for transformation in self.values_transformations]

        attention_scores = [torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) for q, k in zip(queries, keys)]

        if mask is not None:
            attention_scores = [score.masked_fill(mask == 0, float('-inf')) for score in attention_scores]

        attention_probs = [torch.softmax(score, dim=-1) for score in attention_scores]
        # 3. Multiply values by attention weights
        weighted_values = [torch.matmul(prob, v) for prob, v in zip(attention_probs, values)]
        
        # 4. Concatenate results from all attention heads
        concat_weighted_values = torch.cat(weighted_values, dim=-1)
        
        # 5. Pass through fc_out
        output = self.fc_out(concat_weighted_values)
        
        return output


class EncoderBlock(nn.Module):
    def __init__(
            self,
            d_model,
            heads,
            dropout
        ):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(heads, d_model)


    def forward(self, x, src_mask):
        out = self.attention(x, x, x, mask=src_mask)
        out = self.norm1(out + x)

        ff_out = self.feed_forward(out)

        return self.dropout(self.norm2(ff_out + out))


class Encoder(nn.Module):
    def __init__(
            self, 
            d_model,
            heads, 
            max_len,  
            dropout, 
            num_layers, 
            src_vocab_size, 
            device
        ):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, heads, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model
        self.max_len = max_len
        self.device = device

        self.embedding = nn.Embedding(
            src_vocab_size, 
            d_model
        ).to(device=device)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, 
            max_len=max_len
        ).to(device=device)

    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self, 
            d_model, 
            heads, 
            dropout
        ):
        super(DecoderBlock, self).__init__()
        self.attention_masked = Attention(heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.attention = Attention(heads, d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, x_enc, x, trg_mask, src_mask):
        out1 = self.attention(x, x, x, mask=trg_mask)
        out1 = self.norm1(out1 + x)

        out2 = self.attention(queries=out1, keys=x_enc, values=x_enc, mask=src_mask)
        out2 = self.norm2(out1 + out2)

        out3 = self.feed_forward(out2)
        out3 = self.norm3(out2 + out3)

        return self.dropout(out3)


class Decoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            heads, 
            dropout, 
            max_len, 
            num_layers, 
            trg_vocab_size, 
            device
        ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            trg_vocab_size, 
            d_model
        ).to(device=device)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, 
            max_len=max_len
        ).to(device=device)
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, heads, dropout) for _ in range(num_layers)]
        )
        self.d_model = d_model
        self.max_len = max_len
        self.device = device

    def forward(self, x_enc, x, trg_mask, src_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x_enc, x, trg_mask, src_mask)
        return x
    

class Transformer(nn.Module):
    def __init__(
            self, 
            src_pad_idx, 
            trg_pad_idx, 
            device, 
            d_model, 
            heads, 
            dropout, 
            max_len, 
            num_layers, 
            src_vocab_size, 
            trg_vocab_size
        ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            d_model=d_model, 
            heads=heads, 
            dropout=dropout, 
            max_len=max_len, 
            num_layers=num_layers, 
            src_vocab_size=src_vocab_size, 
            device=device
        )
        self.decoder = Decoder(
            d_model=d_model, 
            heads=heads, 
            dropout=dropout, 
            max_len=max_len, 
            num_layers=num_layers, 
            trg_vocab_size=trg_vocab_size, 
            device=device
        )

        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.softmax = nn.Softmax()
        
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(enc_src, trg, trg_mask, src_mask)
        return self.fc_out(out)