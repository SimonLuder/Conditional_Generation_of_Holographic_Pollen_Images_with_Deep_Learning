
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_function(name="silu", inplace=False):
    """
    Get activation function
    :param name: Activation function name
    :param inplace: can optionally do the operation in-place
    :return Activation function
    """
    if name == "relu":
        act = nn.ReLU(inplace=inplace)
    elif name == "relu6":
        act = nn.ReLU6(inplace=inplace)
    elif name == "silu":
        act = nn.SiLU(inplace=inplace)
    elif name == "lrelu":
        act = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "gelu":
        act = nn.GELU()
    else:
        print(f"Unsupported activation function type: {name}")
        act = nn.SiLU(inplace=inplace)
    return act


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, up_weight = ema_params, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old_weight, new_weight):
        if old_weight is None:
            return new_weight
        return old_weight * self.beta + (1 - self.beta) * new_weight

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# class SelfAttention(nn.Module):
#     def __init__(self, channels, size, act="silu"):
#         super(SelfAttention, self).__init__()
#         self.channels = channels
#         self.size = size
#         self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
#         self.ln = nn.LayerNorm(normalized_shape=[channels])
#         self.ff_self = nn.Sequential(
#             nn.LayerNorm(normalized_shape=[channels]),
#             nn.Linear(in_features=channels, out_features=channels),
#             get_activation_function(name=act),
#             nn.Linear(in_features=channels, out_features=channels),
#         )

#     def forward(self, x):
#         print("in", x.shape)
#         B, C, W, H = x.size()
#         x = x.view(B, -1, self.size * self.size).swapaxes(1, 2)
#         x_ln = self.ln(x)
#         attention_value, _ = self.mha(x_ln, x_ln, x_ln)
#         attention_value = attention_value + x
#         attention_value = self.ff_self(attention_value) + attention_value
#         attention_value = attention_value.swapaxes(1, 2).view(B, -1, self.size, self.size)
#         print("out", attention_value.shape)
#         return attention_value


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        query = self.query_conv(x).view(B, -1, W*H).permute(0, 2, 1)
        key = self.key_conv(x).view(B, -1, W*H)
        value = self.value_conv(x).view(B, -1, W*H)

        raw_attention = torch.bmm(query, key)
        attention = torch.softmax(raw_attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma*out + x
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, act="silu"):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.act = act
        self.act_func = get_activation_function(name=self.act)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            get_activation_function(name=self.act),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )

    def forward(self, x):
        if self.residual:
            return self.act_func(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=256, act="silu"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True, act=act),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, act=act),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_channels=256, act="silu"):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True, act=act),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels // 2, act=act),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=emb_channels, out_features=out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
