import torch.nn as nn


class Add_Norm(nn.Module):
    def __init__(self, d_model, dropout, residual, drop_flag=1):
        super(Add_Norm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.residual = residual
        self.drop_flag = drop_flag
        self.dropout = nn.Dropout(dropout) if drop_flag else None

    def forward(self, new, old):
        if self.residual:
            return self.norm(old + new)
        else:
            if self.drop_flag and self.dropout:
                new = self.dropout(new)  # 只对新信息使用 Dropout
            return self.norm(new)


class EncoderLayer(nn.Module):
    def __init__(self, mamba_forward, mamba_backward, d_model=128, d_ff=256, dropout=0.2,
                 activation="relu", bi_dir=0, residual=1):
        super(EncoderLayer, self).__init__()
        self.bi_dir = bi_dir
        self.mamba_forward = mamba_forward
        self.residual = residual
        self.addnorm_for = Add_Norm(d_model, dropout, residual, drop_flag=0)  # 在残差连接中不使用 Dropout

        if self.bi_dir:
            self.mamba_backward = mamba_backward
            self.addnorm_back = Add_Norm(d_model, dropout, residual, drop_flag=0)  # 同样在双向路径中也不使用 Dropout

        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),  # 在 FFN 中使用 Dropout
            nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        )
        self.addnorm_ffn = Add_Norm(d_model, dropout, residual, drop_flag=1)  # FFN 层后可以使用 Dropout

    def forward(self, x):
        # 前向路径
        output_forward = self.mamba_forward(x)
        output_forward = self.addnorm_for(output_forward, x)  # 不在残差连接中使用 Dropout

        if self.bi_dir:
            output_backward = self.mamba_backward(x.flip(dims=[1])).flip(dims=[1])
            output_backward = self.addnorm_back(output_backward, x)
            output = output_forward + output_backward
        else:
            output = output_forward

        temp = output
        output = self.ffn(output.transpose(-1, 1)).transpose(-1, 1)
        output = self.addnorm_ffn(output, temp)  # 在 FFN 之后使用 Dropout
        return output


class Encoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer

    def forward(self, x):
        for mamba_block in self.mamba_layers:
            x = mamba_block(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
