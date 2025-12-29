import torch
import torch.nn as nn
import numpy as np
from BiMamba4TS_layers import Encoder, EncoderLayer
from Embed import PatchEmbedding, TruncateModule
from einops import rearrange
from mamba_ssm import Mamba


class Model(nn.Module):
    def __init__(self, configs, corr=None):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.revin = configs.revin
        self.embed_type = configs.embed_type
        if configs.SRA:
            self.ch_ind = self.SRA(corr, configs.threshold)
        else:
            self.ch_ind = configs.ch_ind

        if configs.seq_len % configs.stride == 0:
            self.patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 1)
            process_layer = nn.Identity()

        self.local_token_layer = PatchEmbedding(configs.seq_len, configs.d_model, configs.patch_len, configs.stride,
                                                configs.dropout, process_layer,
                                                pos_embed_type=None if configs.embed_type in [0,
                                                                                              2] else configs.pos_embed_type,
                                                learnable=configs.pos_learnable,
                                                ch_ind=configs.ch_ind)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=configs.d_model,
                          d_state=configs.d_state,
                          d_conv=configs.d_conv,
                          expand=configs.e_fact,
                          use_fast_path=True),
                    Mamba(d_model=configs.d_model,
                          d_state=configs.d_state,
                          d_conv=configs.d_conv,
                          expand=configs.e_fact,
                          use_fast_path=True),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    bi_dir=configs.bi_dir,
                    residual=configs.residual == 1
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        head_nf = configs.d_model * self.patch_num
        self.head = Flatten_Head(False, configs.enc_in, head_nf, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.revin:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, _, M = x_enc.shape

        enc_out, _ = self.local_token_layer(x_enc.permute(0, 2, 1),
                                            x_mark_enc.permute(0, 2, 1) if self.embed_type == 2 else None)
        if not self.ch_ind:
            enc_out = rearrange(enc_out, '(B M) N D -> (B N) M D', B=B)

        enc_out = self.encoder(enc_out)
        if not self.ch_ind:
            dec_out = rearrange(enc_out, '(B N) M D -> B M N D', B=B)
        else:
            dec_out = rearrange(enc_out, '(B M) N D -> B M N D', B=B)
        dec_out = self.head(dec_out).permute(0, 2, 1)[:, :, :M]

        if self.revin:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, None


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0.2):
        super().__init__()

        self.ch_ind = individual
        self.n_vars = n_vars

        if self.ch_ind:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.ch_ind:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
        return x
