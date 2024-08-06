import time
from audioop import bias
"""
不进行修改
"""
# import thop
# from ptflops import get_model_complexity_info

"""
加上大号dense连接
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.init import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


def window_partition(x, window_size):
    """
     将feature map按照window_size划分成一个个没有重叠的window
     Args:
         x: (B, H, W, C)
         window_size (int): window size(M)

     Returns:
         windows: (num_windows*B, window_size, window_size, C)
     """
    B, H, W, C = x.shape
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
    B, H_p, W_p, C = x.shape
    x = x.view(B, H_p // window_size, window_size, W_p // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    pad_true = bool(pad_b + pad_r + pad_l + pad_t)
    return x, pad_true, H_p, W_p


def window_reverse(windows, window_size, H, W):
    """
        将一个个window还原成一个feature map
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size(M)
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
    """
    # print("H:", H)
    # print("W:", W)
    # print("window shape", windows.shape)

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Cross_Attn(nn.Module):
    def __init__(self, c_dim, num_heads, patch_size):
        super(Cross_Attn, self).__init__()
        self.patch_size = patch_size
        self.q_kv1_proj_weight = nn.Parameter(torch.Tensor(2 * c_dim, c_dim))
        self.k_qv2_proj_weight = nn.Parameter(torch.Tensor(2 * c_dim, c_dim))
        trunc_normal_(self.q_kv1_proj_weight, std=.02)
        trunc_normal_(self.k_qv2_proj_weight, std=.02)
        self.num_heads = num_heads
        self.scale = float(c_dim // num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(c_dim, c_dim)
        mlp_hidden_dim = int(c_dim * 2.)
        self.mlp = Mlp(in_features=c_dim, hidden_features=mlp_hidden_dim, out_features=c_dim, act_layer=nn.GELU)
        self.norm1 = nn.LayerNorm(c_dim)
        self.norm2 = nn.LayerNorm(c_dim)
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, tgt, src):  # tgt: (B,C,H,W)  [tgt_len, batch_size, embed_dim], src: (B,C,H,W)
        B, C, H, W = tgt.shape
        p_z = self.patch_size
        src = src.permute(0, 2, 3, 1)
        shortcut = src
        src = self.norm1(src)
        tgt = self.norm1(tgt.permute(0, 2, 3, 1))
        src, pad_true, H_p, W_p = window_partition(src, p_z)  # src: (B,N_H*N_W, patch, patch, C)
        # -->(B*N, P_h*P_w, C)
        tgt, _, _, _ = window_partition(tgt, p_z)  # tgt: (B, N_H*N_W, patch, patch, C)
        # -->(B*N, P_h*P_w, C)
        q_k_v1 = F.linear(tgt, self.q_kv1_proj_weight).reshape(-1, p_z * p_z, 2, self.num_heads,
                                                               C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_q_v2 = F.linear(src, self.k_qv2_proj_weight).reshape(-1, p_z * p_z, 2, self.num_heads,
                                                               C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_k, v1 = q_k_v1[0], q_k_v1[1]
        k_q, v2 = k_q_v2[0], k_q_v2[1]
        q_cnn = q_k * self.scale
        k_msa = k_q
        attn_cnn = (q_cnn @ k_msa.transpose(-2, -1))  # q: (B,num_heads,H*W,C/num_h), k.T: (B,num_heads,C/num_h,H*W)
        x = (attn_cnn @ v1).transpose(1, 2).flatten(-2)  # attn: (B,num_heads,H*W,H*W), v: (B,num_heads,H*W,C/num_h)
        x_cnn = self.proj(x)

        attn_msa = (attn_cnn.transpose(-2, -1))  # q: (B,num_heads,H*W,C/num_h), k.T: (B,num_heads,C/num_h,H*W)
        x = (attn_msa @ v2).transpose(1, 2).flatten(-2)  # attn: (B,num_heads,H*W,H*W), v: (B,num_heads,H*W,C/num_h)
        x_msa = self.proj(x)

        x_cat = self.alpha * x_cnn + self.beta * x_msa

        x = window_reverse(x_cat, p_z, H_p, W_p)
        if pad_true:
            x = x[:, :H, :W, :].contiguous()
        x = x + shortcut
        x = x + self.mlp(self.norm2(x))  # x: B,H,W,C
        x = x.permute(0, 3, 1, 2)
        return x  # x:(B,C,H,W)


# def calculate_var(input):
#     C = input.size(1)  # B*C*H*W
#
#     def _calculate_channel_var(x):
#         Batch_size = x.size(0)
#         x_var = torch.var(x, dim=(-1, -2))
#         _, b_index = torch.sort(x_var, dim=1, descending=True)
#         x_list = []
#         for i in range(Batch_size):
#             x_list.append(torch.index_select(x[i], 0, b_index[i]).unsqueeze(0))
#         x = torch.cat(x_list)
#         return x[:, :C//2, :, :]
#
#     return _calculate_channel_var(input)  # 取出每个通道，并计算该通道方差
#
#
# def calculate_mean(input):
#     C = input.size(1)  # B*C*H*W
#
#     def _calculate_channel_mean(x):
#         Batch_size = x.size(0)
#         x_mean = torch.mean(x, dim=(-1, -2))
#         _, b_index = torch.sort(x_mean, dim=1, descending=True)
#         x_list = []
#         for i in range(Batch_size):
#             x_list.append(torch.index_select(x[i], 0, b_index[i]).unsqueeze(0))
#         x = torch.cat(x_list)
#         return x[:, :C//2, :, :]
#
#     return _calculate_channel_mean(input)

def calculate_var(input):
    def _calculate_channel_var(x):
        return torch.var(x, dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

    return _calculate_channel_var(input)  # 取出每个通道，并计算该通道方差


def calculate_mean(input):
    def _calculate_channel_mean(x):
        return torch.mean(x, dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

    return _calculate_channel_mean(input)


class Dconv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, (1, 1), padding=padding)
        self.conv2 = nn.Conv2d(in_dim, out_dim, (kernel_size, kernel_size), padding=padding, groups=in_dim)

    def forward(self, input):
        out = self.conv2(self.conv1(input))
        return out


class Conv_Gelu_Res(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding='same'):
        super().__init__()
        self.conv1 = Dconv(in_dim, out_dim, kernel_size, padding)
        self.act = nn.GELU()

    def forward(self, input):
        out = self.act(self.conv1(input) + input)
        return out


# # Gated-Dconv Feed-Forward Network (GDFN)
# class GDFN(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(GDFN, self).__init__()
#         self.dim = dim
#         hidden_features = int(dim * ffn_expansion_factor)
#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias, padding='same')
#         self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
#                                 groups=hidden_features * 2, bias=bias)
#         self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias, padding='same')
#
#     def forward(self, x):
#         short = x
#         x = self.project_in(x)
#         x1, x2 = self.dwconv(x).chunk(2, dim=1)
#         x = F.gelu(x1) * x2
#         x = self.project_out(x) + short
#         return x


class Channel_Atteneion(nn.Module):
    def __init__(self, c_dim, reduction):
        super().__init__()
        self.conv_mean = calculate_mean
        self.conv_variance = calculate_var
        self.after_mean = nn.Sequential(
            nn.Conv2d(c_dim, c_dim // reduction, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(c_dim // reduction, c_dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # self.after_var = nn.Sequential(
        #     nn.Conv2d(c_dim, c_dim // reduction, 1, padding=0, bias=True),
        #     nn.GELU(),
        #     nn.Conv2d(c_dim // reduction, c_dim, 1, padding=0, bias=True),
        #     nn.Sigmoid()
        # )
        # self.alpha = nn.Parameter(torch.zeros(1, c_dim, 1, 1))
        # trunc_normal_(self.alpha, std=.02)
        # self.beta = nn.Parameter(torch.zeros(1, c_dim, 1, 1))
        # trunc_normal_(self.beta, std=.02)
        # self.conv_after_concat = nn.Conv2d(c_dim * 2, c_dim, 1)

    def forward(self, x):
        channel_mean = self.after_mean(self.conv_mean(x))
        # # x_mean = channel_mean * self.alpha
        x_mean = channel_mean * x
        # channel_var = self.after_var(self.conv_variance(x))
        # x_var = channel_var * self.beta
        # x_var = channel_var * x
        # x = self.conv_after_concat(torch.cat((x_mean, x_var), 1))
        # x = torch.cat((x_mean, x_var), 1)
        return x_mean


class Res_Channel_Attn(nn.Module):
    def __init__(self, c_dim, reduction):
        super().__init__()
        modules_body = []
        self.input = input
        self.conv_1 = Conv_Gelu_Res(c_dim, c_dim, 3, padding='same')
        modules_body.append(self.conv_1)
        self.ca = Channel_Atteneion(c_dim, reduction)
        modules_body.append(self.ca)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        ca_x = self.body(x)
        ca_x += x
        return ca_x

    def flops(self):
        flops = 0
        flops += self.conv_1.flops()
        flops += self.ca.flops()
        return flops


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qk = self.qk(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        v = self.v(y).reshape(B_, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ration = mlp_ratio
        "shift_size must in 0-window_size"
        assert 0 <= self.shift_size < self.window_size
        "层归一化"
        # self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size),
                                    num_heads=self.num_heads,
                                    qkv_bias=qkv_bias,
                                    proj_drop=drop)
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask):  # x: B,C,H,W

        B, C_2, H, W = x.shape
        C = int(C_2/2)
        x = x.permute(0, 2, 3, 1)  # x: B,H,W,2*C
        _, shortcut = torch.chunk(x, 2, dim=3)
        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape


        if self.shift_size > 0:
            shift_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        else:
            shift_x = x
            mask = None

        shift_x, shift_y = torch.chunk(shift_x, 2, dim=3)
        shift_x = self.norm1(shift_x)
        # 划分窗口
        x_window = window_partition(shift_x, window_size=self.window_size)[0]
        y_window = window_partition(shift_y, window_size=self.window_size)[0]
        x_window = x_window.view(-1, self.window_size * self.window_size, C)
        y_window = y_window.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA mask: None/其他 计算注意力并且加上掩码
        attn_x = self.attn(x_window, y_window, mask)

        # 从窗口还原到原来的大小
        attn_x = attn_x.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        x_reverse = window_reverse(attn_x, self.window_size, Hp, Wp)  # [B, H', W', C]

        # shift 还原
        if self.shift_size > 0:
            x = torch.roll(x_reverse, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = x_reverse

        if pad_r or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            shortcut = shortcut[:, :H, :W, :].contiguous()
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))  # x: B,H,W,C
        x = x.permute(0, 3, 1, 2)
        return x  # x: B,C,H,W


class Spatial_Attn(nn.Module):
    def __init__(self, c_dim, depth, windows_size, num_heads):
        super().__init__()
        self.dwconv1 = nn.Conv2d(c_dim, c_dim, 1, 1, padding='same', padding_mode='reflect', groups=c_dim)
        self.dwconv3 = nn.Conv2d(c_dim, c_dim, 3, 1, padding='same', padding_mode='reflect', groups=c_dim)
        self.dwconv5 = nn.Conv2d(c_dim, c_dim, 5, 1, padding='same', padding_mode='reflect', groups=c_dim)
        # self.dwconv7 = nn.Conv2d(c_dim, c_dim, 7, 1, padding='same', padding_mode='reflect', groups=c_dim)

        # self.dilaconv1 = nn.Conv2d(c_dim, c_dim, 3, 1, padding='same', padding_mode='reflect', dilation=1,
        #                            groups=int(c_dim / 6))
        # self.dilaconv2 = nn.Conv2d(c_dim, c_dim, 3, 1, padding='same', padding_mode='reflect', dilation=2,
        #                            groups=int(c_dim / 6))
        # self.dilaconv3 = nn.Conv2d(c_dim, c_dim, 3, 1, padding='same', padding_mode='reflect', dilation=3,
        #                            groups=int(c_dim / 6))

        # self.conv1x1_1 = nn.Conv2d(c_dim, c_dim, 1)
        # self.conv1x1_2 = nn.Conv2d(c_dim, c_dim, 1)
        # self.conv1x1_3 = nn.Conv2d(c_dim, c_dim, 1)
        self.act = nn.GELU()
        # self.att_act = nn.Sigmoid()
        # self.conv1x1_end = nn.Conv2d(4 * c_dim, c_dim, 1)
        # self.conv3x3_end = nn.Conv2d(c_dim, c_dim, 3, padding='same', padding_mode='reflect', groups=c_dim)
        swin_body = []
        self.window_size = windows_size
        for i in range(depth):
            if i % 2:
                shift_size = windows_size // 2
            else:
                shift_size = 0
            self.shift_size = shift_size
            swin_body.append(SwinTransformerBlock(c_dim, num_heads, window_size=windows_size, shift_size=shift_size,
                                                  mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                                                  act_layer=nn.GELU))
        self.swin_body = nn.Sequential(*swin_body)

    def creat_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size) * self.window_size)
        Wp = int(np.ceil(W / self.window_size) * self.window_size)

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_s = (slice(0, -self.window_size),
               slice(-self.window_size, -self.shift_size),
               slice(-self.shift_size, None))
        w_s = (slice(0, -self.window_size),
               slice(-self.window_size, -self.shift_size),
               slice(-self.shift_size, None))
        c = 0
        for h in h_s:
            for w in w_s:
                img_mask[:, h, w, :] = c
                c += 1
        mask_window = window_partition(img_mask, self.window_size)[0]  # [nW, Mh, Mw, 1]
        mask_window = mask_window.view(-1, self.window_size * self.window_size)
        mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)
        # [nW, Mh*Mw, Mh*Mw]
        mask = mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))
        return mask

    def forward(self, x):
        src = x
        dw1 = self.dwconv1(x)
        dw3 = self.dwconv3(x)
        dw5 = self.dwconv5(x)
        # dw7 = self.dwconv7(x)
        feat = self.act(dw1 + dw3 + dw5 + x)
        _, _, H, W, = x.shape
        mask = self.creat_mask(x, H, W)
        for body in self.swin_body:
            src = body(torch.cat([feat, src], dim=1), mask)
        info_mix = src

        return info_mix


class Res_Spatial_Attn(nn.Module):
    def __init__(self, c_dim, depth, windows_size, num_heads):
        super().__init__()
        modules_body = []
        modules_body.append(Conv_Gelu_Res(c_dim, c_dim, 3, padding='same'))
        modules_body.extend([Spatial_Attn(c_dim, depth, windows_size, num_heads)])
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def Pixelshuffle_Block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), (kernel_size, kernel_size), (stride, stride),
                     padding='same')
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


class BasicLayer(nn.Module):
    def __init__(self, c_dim, reduction, RC_depth, RS_depth, depth, windows_size, num_heads):
        super(BasicLayer, self).__init__()
        self.body_0 = []
        self.body_1 = []
        for i_layer in range(RC_depth):
            layer = Res_Channel_Attn(c_dim, reduction)
            self.body_0.append(layer)
        dwconv_0 = Dconv(c_dim, c_dim, 3, padding='same')
        self.body_0.append(dwconv_0)
        for i_layer in range(RS_depth):
            layer = Res_Spatial_Attn(c_dim, depth, windows_size, num_heads)
            self.body_1.append(layer)
        dwconv_1 = Dconv(c_dim, c_dim, 3, padding='same')
        self.body_1.append(dwconv_1)
        self.res_channel_attn = nn.Sequential(*self.body_0)
        self.res_spatial_attn = nn.Sequential(*self.body_1)
        self.conv1x1_1 = nn.Conv2d(c_dim, c_dim // 2, 1)
        self.conv1x1_2 = nn.Conv2d(c_dim, c_dim // 2, 1)
        self.conv1x1_3 = nn.Conv2d(c_dim * 2, c_dim, 1)

    def forward(self, x):
        short_cut = x
        res1 = self.conv1x1_1(x)
        x = self.res_channel_attn(x) + short_cut
        res2 = self.conv1x1_2(x)
        x = self.res_spatial_attn(x) + x
        out_B = self.conv1x1_3(torch.cat([res1, res2, x], dim=1))
        out_lr = out_B + short_cut
        return out_lr

# @ARCH_REGISTRY.register()
class SpcT_Tiny(nn.Module):
    def __init__(self, rgb_mean=(0.4488, 0.4371, 0.4040), upscale_factor=4, c_dim=42, reduction=16, Bsc_depth=4, RS_depth=2, RC_depth=4, depth=2,
                 windows_size=8, num_heads=3):
        super(SpcT_Tiny, self).__init__()
        self.body = []
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv_shallow = nn.Conv2d(3, c_dim, kernel_size=(3, 3), stride=(1, 1), padding='same')
        for i_layer in range(Bsc_depth):
            layer = BasicLayer(c_dim, reduction, RC_depth, RS_depth, depth, windows_size, num_heads)
            self.body.append(layer)
        self.conv_before_upsample = Dconv(c_dim, c_dim, 3, padding='same')
        self.upsample = Pixelshuffle_Block(c_dim, 3, upscale_factor=upscale_factor, kernel_size=3)
        self.bsc_layer = nn.Sequential(*self.body)
        self.c = nn.Conv2d(Bsc_depth * c_dim, c_dim, 1)
        # self.conv1x1_1 = nn.Conv2d(c_dim, c_dim, 1)
        # self.conv1x1_2 = nn.Conv2d(c_dim, c_dim, 1)
        # self.conv1x1_3 = nn.Conv2d(c_dim, c_dim, 1)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = x - self.mean
        out_fea = self.conv_shallow(x)
        x1 = self.bsc_layer[0](out_fea)
        x2 = self.bsc_layer[1](x1)
        x3 = self.bsc_layer[2](x2)
        x4 = self.bsc_layer[3](x3)
        out_B = self.c(torch.cat([x1, x2, x3, x4], dim=1))
        out_lr = self.conv_before_upsample(out_B) + out_fea

        output = self.upsample(out_lr) + self.mean

        return output

if __name__ == '__main__':

    swin_lap = SpcT_Tiny(rgb_mean=(0.4488, 0.4371, 0.4040))
    total_params = sum(p.numel() for p in swin_lap.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in swin_lap.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    x = torch.randn((1, 3, 320, 180))
    swin_lap.cuda()
    out = swin_lap(x.cuda())
    print(out.shape)
